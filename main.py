import subprocess
import sys
import pickle # Import the pickle module


import numpy as np
import collections
import json
import os
import google.generativeai.protos as protos
from google.protobuf import struct_pb2
from google.protobuf import json_format
from proto.marshal.collections.maps import MapComposite
from typing import Dict, Any, List, Optional
import re
import random
import datetime

# Install google-generativeai if not already installed
try:
    import google.generativeai as genai
except ImportError:
    print("Installing google-generativeai...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
    import google.generativeai as genai


from google.generativeai.types import generation_types
import matplotlib.pyplot as plt
import scipy.stats as stats
from config import  NUM_TRIALS, POWER_PER_20MHZ_CARRIER_W, REFERENCE_BANDWIDTH_MHZ, REFERENCE_ENERGY_FOR_SAVINGS_W, SLA_LATENCY_THRESHOLD_MS, NUM_TIME_STEPS

import e2_api_tool
import network_simulator
from digital_twin import DigitalTwin
from collective_memory import CollectiveMemory
from a2a import A2ANegotiationManager
from agents import RanAgent, EdgeAgent





all_queried_strategies_ages = []
total_successful_strategies_queried = 0
total_failed_strategies_queried = 0

def _run_single_scenario(num_trials: int, scenario_name: str, memory_enabled: bool, debiasing_enabled: bool, trial_start_steps: List[int]):
    # Reset global counters for each scenario run
    all_queried_strategies_ages.clear() # Corrected: Modify list in place
    global total_successful_strategies_queried
    global total_failed_strategies_queried
    total_successful_strategies_queried = 0
    total_failed_strategies_queried = 0

    scenario_results = []
    scenario_excess_latencies = [] # New list to store excess latencies for ALESA
    scenario_conflicts_raw = [] # Raw boolean/int for conflicts per trial
    scenario_sla_violations_raw = [] # Raw boolean/int for SLA violations per trial
    
    # Pass ETR-DDS parameters to CollectiveMemory
    collective_memory = CollectiveMemory(
        debiasing_enabled=debiasing_enabled,
        alpha=1.0, beta=0.5, gamma=0.8, delta=1.0, # Example values, can be tuned
        decay_rate_factor=80.0
    ) if memory_enabled else None

    print(f"\n\n======== Running Scenario: {scenario_name} (across {num_trials} trials) ========")

    # Lists to collect raw data for plotting
    scenario_latencies = []
    scenario_energy_savings = []
    scenario_consensus_times = []
    scenario_queried_ages = [] # To store ages for this specific scenario

    total_average_energy_across_trials = 0.0
    total_average_latency_across_trials = 0.0
    

    for trial_num in range(num_trials):
        start_time_step = trial_start_steps[trial_num] # Use pre-generated start time step
        print(f"\n--- Scenario: {scenario_name} - Trial {trial_num + 1} (Starting at time step {start_time_step}) ---")
        
        # Reset simulation and generate new traffic for each trial
        e2_api_tool.simulator.reset_simulation(start_time_step=start_time_step)

        # Collect simulator parameters for the DigitalTwin
        simulator_params_for_dt = {
            'tau': e2_api_tool.simulator.tau,
            'U': e2_api_tool.simulator.U, # Pass the updated U
            'fmax': e2_api_tool.simulator.fmax,
            'min_spectral_efficiency': e2_api_tool.simulator.min_spectral_efficiency,
            'max_spectral_efficiency': e2_api_tool.simulator.max_spectral_efficiency,
            'POWER_PER_20MHZ_CARRIER_W': POWER_PER_20MHZ_CARRIER_W,
            'REFERENCE_BANDWIDTH_MHZ': REFERENCE_BANDWIDTH_MHZ,
            'REFERENCE_ENERGY_FOR_SAVINGS_W': REFERENCE_ENERGY_FOR_SAVINGS_W,
            '_TRAFFIC_BASE_SLICE': e2_api_tool.simulator._TRAFFIC_BASE_SLICE,
            '_TRAFFIC_VARIATION_SLICE': e2_api_tool.simulator._TRAFFIC_VARIATION_SLICE
        }
        digital_twin_instance = DigitalTwin(simulator_params_for_dt)


        ran_agent_tools = [e2_api_tool.get_metrics, e2_api_tool.enforce_actions]
        edge_agent_tools = [e2_api_tool.get_metrics, e2_api_tool.enforce_actions]

        if memory_enabled:
            ran_agent_tools.append(collective_memory.query_memory)
            edge_agent_tools.append(collective_memory.query_memory)

        ran_agent = RanAgent(
            max_ran_bw=e2_api_tool.MAX_RAN_BW_MHZ, # Pass max_ran_bw here
            tools=ran_agent_tools,
            collective_memory_tool=collective_memory if memory_enabled else None,
            digital_twin_instance=digital_twin_instance, # Pass the new tool
            debiased_memory_prompt_enabled=debiasing_enabled # Pass this flag
        )
        edge_agent = EdgeAgent(
            max_ran_bw=e2_api_tool.MAX_RAN_BW_MHZ, # Pass max_ran_bw here
            tools=edge_agent_tools,
            collective_memory_tool=collective_memory if memory_enabled else None,
            digital_twin_instance=digital_twin_instance, # Pass the new tool
            debiased_memory_prompt_enabled=debiasing_enabled # Pass this flag
        )
        
        negotiation_manager = A2ANegotiationManager(ran_agent, edge_agent, e2_api_tool, collective_memory, max_iterations=8, trial_num=trial_num) # Pass trial_num here
        
        trial_result = negotiation_manager.run_negotiation()
        scenario_results.append(trial_result)
        print(f"Trial {trial_num + 1} Summary: Consensus Time = {trial_result['consensus_time']}, Unresolved Negotiation = {trial_result['unresolved_negotiation']}, SLA Violation = {trial_result['sla_violation_occurred']}, Parsing Failure = {trial_result.get('unparseable_message_failure', False)}")
        
        # Collect data for plotting
        scenario_latencies.append(trial_result.get("average_latency_this_trial", 0.0))
        if not trial_result["unresolved_negotiation"]: # Only include energy savings for successful agreements
            scenario_energy_savings.append(trial_result.get("saved_energy_percent", 0.0))
            scenario_consensus_times.append(trial_result.get("consensus_time", 0))
        else: # For unresolved, consensus time is max_iterations + 1
            scenario_consensus_times.append(trial_result.get("consensus_time", negotiation_manager.max_iterations + 1))

        # Calculate and store excess latency for ALESA
        if trial_result["sla_violation_occurred"]:
            excess = trial_result["average_latency_this_trial"] - SLA_LATENCY_THRESHOLD_MS
            scenario_excess_latencies.append(excess)

        # Collect raw conflict and SLA violation data for CDFs
        # Only count as a conflict if it's unresolved AND NOT due to a parsing failure
        scenario_conflicts_raw.append(1 if trial_result["unresolved_negotiation"] and not trial_result.get("unparseable_message_failure", False) else 0)
        # Only count SLA violations if they are NOT due to a parsing failure
        scenario_sla_violations_raw.append(1 if trial_result["sla_violation_occurred"] and not trial_result.get("unparseable_message_failure", False) else 0)


        # Use the final energy and latency directly from the trial result
        total_average_energy_across_trials += trial_result.get("average_energy_this_trial", 0.0)
        total_average_latency_across_trials += trial_result.get("average_latency_this_trial", 0.0)

    # After all trials for this scenario, add the collected ages
    scenario_queried_ages.extend(all_queried_strategies_ages)


    print(f"\n======== Scenario '{scenario_name}' Completed ========")
    
    total_consensus_time = 0
    num_agreements = 0
    
    # KPIs
    num_unresolved_negotiations_kpi = 0 # This will be "Number of Conflicts"
    total_sla_violations_actual = 0 # This will be "SLA Violation Rate"

    for result in scenario_results:
        # Consensus time and agreement rate calculation remains for successful agreements
        if not result["unresolved_negotiation"]:
            total_consensus_time += result["consensus_time"]
            num_agreements += 1
        else:
            total_consensus_time += result["consensus_time"] # Still add to total consensus time even if unresolved
            # Only count as a conflict if it's unresolved AND NOT due to a parsing failure
            if not result.get("unparseable_message_failure", False):
                num_unresolved_negotiations_kpi += 1 # Increment conflicts if unresolved and not parsing failure

        # Only count SLA violations if they are NOT due to a parsing failure
        if result["sla_violation_occurred"] and not result.get("unparseable_message_failure", False):
            total_sla_violations_actual += 1


    average_consensus_time = total_consensus_time / num_trials if num_trials > 0 else 0.0
    number_of_conflicts_kpi = num_unresolved_negotiations_kpi # Corrected KPI
    agreement_rate = (num_agreements / num_trials) * 100 if num_trials > 0 else 0.0
    sla_violation_rate = (total_sla_violations_actual / num_trials) * 100 if num_trials > 0 else 0.0 # Corrected KPI

    average_energy_saving_successful = 0.0
    if num_agreements > 0:
        total_energy_savings_sum = sum(r["saved_energy_percent"] for r in scenario_results if not r["unresolved_negotiation"])
        average_energy_saving_successful = total_energy_savings_sum / num_agreements

    avg_age_queried_strategies = 0.0
    std_age_queried_strategies = 0.0
    if scenario_queried_ages: # Use scenario_queried_ages for this specific scenario
        avg_age_queried_strategies = np.mean(scenario_queried_ages)
        std_age_queried_strategies = np.std(scenario_queried_ages)

    # Calculate Average Latency Exceeding SLA (ALESA)
    average_alesa = np.mean(scenario_excess_latencies) if scenario_excess_latencies else 0.0

    # New overall averages
    overall_avg_energy = total_average_energy_across_trials / num_trials if num_trials > 0 else 0.0
    overall_avg_latency = total_average_latency_across_trials / num_trials if num_trials > 0 else 0.0

    # Calculate Ratio of Successful to Failed Strategies Queried (Numeric for plotting)
    ratio_success_to_failed_numeric = np.nan
    if total_successful_strategies_queried > 0 and total_failed_strategies_queried > 0:
        ratio_success_to_failed_numeric = total_successful_strategies_queried / total_failed_strategies_queried
    elif total_successful_strategies_queried > 0 and total_failed_strategies_queried == 0:
        ratio_success_to_failed_numeric = float('inf') # Represent Infinity numerically
    elif total_successful_strategies_queried == 0 and total_failed_strategies_queried > 0:
        ratio_success_to_failed_numeric = 0.0 # Represent 0.00 numerically


    metrics = {
        "Average Agentic Consensus Time": f"{average_consensus_time:.2f} iterations",
        "Number of Conflicts (Unresolved Negotiations)": number_of_conflicts_kpi,
        "Agreement Rate": f"{agreement_rate:.2f}%",
        "SLA Violation Rate": f"{sla_violation_rate:.2f}%",
        "Average Latency Exceeding SLA (ALESA)": f"{average_alesa:.2f} ms", # New metric
        "Average Energy Savings (for successful agreements)": f"{average_energy_saving_successful:.2f}%",
        "Average Age of Queried Strategies (in time steps)": f"{avg_age_queried_strategies:.2f}" if scenario_queried_ages else "N/A",
        "Std Dev Age of Queried Strategies (in time steps)": f"{std_age_queried_strategies:.2f}" if scenario_queried_ages else "N/A",
        "Overall Average Energy Consumption (W)": f"{overall_avg_energy:.2f}",
        "Overall Average Latency (ms)": f"{overall_avg_latency:.2f}",
        "Ratio of Successful to Failed Strategies Queried": f"{ratio_success_to_failed_numeric:.2f}" if not np.isnan(ratio_success_to_failed_numeric) and not np.isinf(ratio_success_to_failed_numeric) else ("Infinity" if np.isinf(ratio_success_to_failed_numeric) else "N/A"),
        "Ratio of Successful to Failed Strategies Queried (Numeric)": ratio_success_to_failed_numeric # Store numeric for plotting
    }

    print(f"\n--- Performance Metrics for Scenario: {scenario_name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return {
        "metrics": metrics,
        "latencies": scenario_latencies,
        "energy_savings": scenario_energy_savings,
        "consensus_times": scenario_consensus_times,
        "queried_ages": scenario_queried_ages, # Return the collected ages for this scenario
        "alesa": scenario_excess_latencies, # Return raw excess latencies for CDF if needed, or just the average
        "conflicts_raw": scenario_conflicts_raw, # Return raw conflict data
        "sla_violations_raw": scenario_sla_violations_raw # Return raw SLA violation data
    }

def plot_cdf(data_dict: Dict[str, List[float]], title: str, xlabel: str, ylabel: str, filename: str):
    plt.figure(figsize=(10, 8))
    for label, data in data_dict.items():
        if not data:
            continue
        sorted_data = np.sort(data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, y, marker='.', linestyle='-', label=label)
        
        median_val = np.median(data)
        plt.axvline(x=median_val, color=plt.gca().lines[-1].get_color(), linestyle='--', label=f'{label} Median: {median_val:.2f}')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_bar_chart(data_dict: Dict[str, float], title: str, ylabel: str, filename: str):
    scenarios = list(data_dict.keys())
    values = list(data_dict.values())

    plt.figure(figsize=(10, 8))
    bars = plt.bar(scenarios, values, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title(title)
    plt.xlabel("Scenario")
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_age_of_strategies(data_dict: Dict[str, Dict[str, float]], title: str, ylabel: str, filename: str):
    scenarios = list(data_dict.keys())
    means = [data_dict[s]['mean'] for s in scenarios]
    stds = [data_dict[s]['std'] for s in scenarios]

    plt.figure(figsize=(10, 8))
    bars = plt.bar(scenarios, means, yerr=stds, capsize=5, color=['lightcoral', 'lightgreen'])
    plt.title(title)
    plt.xlabel("Memory Scenario")
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)

    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.5, f"Avg: {means[i]:.2f}\nStd: {stds[i]:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def run_all_scenarios(num_trials: int = 100): # Changed num_trials to 100
    print("Starting all negotiation scenarios...")

    # Set a fixed random seed for reproducibility across all scenarios
    np.random.seed(42)
    random.seed(42)

    scenario_raw_data = {}
    scenario_summary_metrics = {}

    # Generate a pool of unique starting time steps (representing distinct traffic patterns)
    # The size of this pool is NUM_TIME_STEPS (8 in this case)
    unique_start_steps_pool = list(range(NUM_TIME_STEPS))
    random.shuffle(unique_start_steps_pool) # Shuffle to randomize the initial order

    # For each trial, randomly select a start_time_step from this pool with replacement
    # This creates non-uniform repetitions of contexts
    fixed_trial_start_steps = [random.choice(unique_start_steps_pool) for _ in range(num_trials)]


    # Scenario 1: Without memory
    results_no_memory = _run_single_scenario(
        num_trials=num_trials,
        scenario_name="w/o memory",
        memory_enabled=False,
        debiasing_enabled=False,
        trial_start_steps=fixed_trial_start_steps
    )
    scenario_raw_data["w/o memory"] = {
        "latencies": results_no_memory["latencies"],
        "energy_savings": results_no_memory["energy_savings"],
        "consensus_times": results_no_memory["consensus_times"],
        "queried_ages": results_no_memory["queried_ages"],
        "alesa": results_no_memory["alesa"], # Store raw excess latencies
        "conflicts_raw": results_no_memory["conflicts_raw"], # Store raw conflict data
        "sla_violations_raw": results_no_memory["sla_violations_raw"] # Store raw SLA violation data
    }
    scenario_summary_metrics["w/o memory"] = results_no_memory["metrics"]


    # Scenario 2: With collective memory without debiasing
    results_memory_no_debias = _run_single_scenario(
        num_trials=num_trials,
        scenario_name="w/ memory",
        memory_enabled=True,
        debiasing_enabled=False, # This is now correctly set to False for vanilla memory
        trial_start_steps=fixed_trial_start_steps
    )
    scenario_raw_data["w/ memory"] = {
        "latencies": results_memory_no_debias["latencies"],
        "energy_savings": results_memory_no_debias["energy_savings"],
        "consensus_times": results_memory_no_debias["consensus_times"],
        "queried_ages": results_memory_no_debias["queried_ages"],
        "alesa": results_memory_no_debias["alesa"], # Store raw excess latencies
        "conflicts_raw": results_memory_no_debias["conflicts_raw"], # Store raw conflict data
        "sla_violations_raw": results_memory_no_debias["sla_violations_raw"] # Store raw SLA violation data
    }
    scenario_summary_metrics["w/ memory"] = results_memory_no_debias["metrics"]


    # Scenario 3: Collective memory with debiasing mechanisms
    results_memory_with_debias = _run_single_scenario(
        num_trials=num_trials,
        scenario_name="w/ unbiased memory",
        memory_enabled=True,
        debiasing_enabled=True, # This is correctly set to True for ETR-DDS
        trial_start_steps=fixed_trial_start_steps
    )
    scenario_raw_data["w/ unbiased memory"] = {
        "latencies": results_memory_with_debias["latencies"],
        "energy_savings": results_memory_with_debias["energy_savings"],
        "consensus_times": results_memory_with_debias["consensus_times"],
        "queried_ages": results_memory_with_debias["queried_ages"],
        "alesa": results_memory_with_debias["alesa"], # Store raw excess latencies
        "conflicts_raw": results_memory_with_debias["conflicts_raw"], # Store raw conflict data
        "sla_violations_raw": results_memory_with_debias["sla_violations_raw"] # Store raw SLA violation data
    }
    scenario_summary_metrics["w/ unbiased memory"] = results_memory_with_debias["metrics"]


    print("\n\n======== Comparative Performance Metrics Across Scenarios ========")
    for scenario_name, metrics in scenario_summary_metrics.items():
        print(f"\n--- {scenario_name} ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    return scenario_raw_data, scenario_summary_metrics

if __name__ == "__main__":
    results_file_name = 'simulation_results.pkl'

    if os.path.exists(results_file_name):
        print(f"Loading simulation results from {results_file_name}...")
        with open(results_file_name, 'rb') as f:
            raw_data, summary_metrics = pickle.load(f)
        print("Results loaded successfully.")
    else:
        print(f"Simulation results file {results_file_name} not found. Running full simulation...")
        raw_data, summary_metrics = run_all_scenarios(num_trials=NUM_TRIALS) # Run with 100 trials
        print(f"Saving simulation results to {results_file_name}...")
        with open(results_file_name, 'wb') as f:
            pickle.dump((raw_data, summary_metrics), f)
        print("Results saved successfully.")

    # 1. Figure with latency CDF comparison
    latency_data_for_plot = {
        s: raw_data[s]["latencies"] for s in raw_data
    }
    plot_cdf(latency_data_for_plot, "Latency CDF Comparison", "Latency (ms)", "CDF", "latency_cdf.png")

    # 2. Figure with energy saving CDF comparison
    energy_saving_data_for_plot = {
        s: raw_data[s]["energy_savings"] for s in raw_data if s != "w/o memory" # Only memory scenarios have energy savings for successful agreements
    }
    plot_cdf(energy_saving_data_for_plot, "Energy Savings CDF Comparison (Successful Agreements)", "Energy Savings (%)", "CDF", "energy_savings_cdf.png")

    # 3. Figure with Average and Std Dev of Age of Queried Strategies
    age_of_strategies_data = {
        s: {"mean": np.mean(raw_data[s]["queried_ages"]) if raw_data[s]["queried_ages"] else 0,
            "std": np.std(raw_data[s]["queried_ages"]) if raw_data[s]["queried_ages"] else 0}
        for s in raw_data if raw_data[s]["queried_ages"] # Only for scenarios where strategies were queried
    }
    plot_age_of_strategies(age_of_strategies_data, "Average and Std Dev of Age of Queried Strategies", "Age (time steps)", "age_of_strategies.png")

    # 4. Number of conflicts (unresolved) vs scenarios
    conflicts_data = {
        s: summary_metrics[s]["Number of Conflicts (Unresolved Negotiations)"]
        for s in summary_metrics
    }
    plot_bar_chart(conflicts_data, "Number of Conflicts (Unresolved Negotiations) vs Scenarios", "Number of Conflicts", "conflicts_bar.png")

    # 5. SLA violation rate vs scenarios
    sla_violation_data = {
        s: float(summary_metrics[s]["SLA Violation Rate"].replace('%', ''))
        for s in summary_metrics
    }
    plot_bar_chart(sla_violation_data, "SLA Violation Rate vs Scenarios", "SLA Violation Rate (%)", "sla_violation_bar.png")

    # 6. CDF and median of Agentic Consensus Time vs scenarios
    consensus_time_data_for_plot = {
        s: raw_data[s]["consensus_times"] for s in raw_data
    }
    plot_cdf(consensus_time_data_for_plot, "Agentic Consensus Time CDF Comparison", "Consensus Time (iterations)", "CDF", "consensus_time_cdf.png")

    # 7. Ratio of Successful to Failed Strategies Queried vs scenarios (to tackle confirmation bias)
    ratio_data = {
        s: summary_metrics[s]["Ratio of Successful to Failed Strategies Queried (Numeric)"]
        for s in summary_metrics
    }
    
    # Filter out infinite values for plotting, as they distort bar charts
    filtered_ratio_data = {k: v for k, v in ratio_data.items() if not np.isinf(v) and not np.isnan(v)}
    plot_bar_chart(filtered_ratio_data, "Ratio of Successful to Failed Strategies Queried vs Scenarios", "Ratio (Success/Failed)", "success_failed_ratio_bar.png")

    # 8. Average Latency Exceeding SLA (ALESA) vs scenarios (New Plot)
    alesa_data = {
        s: float(summary_metrics[s]["Average Latency Exceeding SLA (ALESA)"].replace(' ms', ''))
        for s in summary_metrics
    }
    plot_bar_chart(alesa_data, "Average Latency Exceeding SLA (ALESA) vs Scenarios", "ALESA (ms)", "alesa_bar.png")

    # NEW PLOTS: CDFs for Conflicts and SLA Violations
    cumulative_conflicts_data = {
        s: np.cumsum(raw_data[s]["conflicts_raw"]).tolist() for s in raw_data
    }
    plot_cdf(cumulative_conflicts_data, "Cumulative Conflicts CDF Comparison", "Cumulative Conflicts", "CDF", "cumulative_conflicts_cdf.png")

    cumulative_sla_violations_data = {
        s: np.cumsum(raw_data[s]["sla_violations_raw"]).tolist() for s in raw_data
    }
    plot_cdf(cumulative_sla_violations_data, "Cumulative SLA Violations CDF Comparison", "Cumulative SLA Violations", "CDF", "cumulative_sla_violations_cdf.png")

    # Also, consider a separate print statement for scenarios with "Infinity" ratio, if any.
    infinity_ratio_scenarios = [s for s, v in ratio_data.items() if np.isinf(v)]
    if infinity_ratio_scenarios:
        print(f"\nNote: The 'Ratio of Successful to Failed Strategies Queried' for the following scenarios was 'Infinity' (no failed strategies queried) and is not shown on the bar chart: {', '.join(infinity_ratio_scenarios)}")
