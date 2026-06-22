import re
import json
import numpy as np
from typing import Dict, Any, List
from config import SLA_LATENCY_THRESHOLD_MS
import e2_api_tool
import network_simulator
# A global list to track ages of queried strategies for analysis.
# This approach is simple for a single-threaded simulation.
# In a more complex application, this might be handled by a dedicated analytics service.
all_queried_strategies_ages = []
total_successful_strategies_queried = 0
total_failed_strategies_queried = 0


class CollectiveMemory:
    def __init__(self, debiasing_enabled: bool = True,
                 alpha: float = 1.0, beta: float = 0.5, gamma: float = 1.0, delta: float = 0.5,
                 decay_rate_factor: float = 1.0): # smaller decay factor to encourage older memories as well
        self.episodic_logs: List[Dict[str, Any]] = []
        self.distilled_strategies: List[Dict[str, Any]] = []
        self.debiasing_enabled = debiasing_enabled
        
        # ETR-DDS parameters
        self.alpha = alpha      # Weight for semantic similarity
        self.beta = beta        # Weight for time-decay
        self.gamma = gamma      # Weight for diversity penalty (higher means more penalty for similarity)
        self.delta = delta      # Weight for inflection bonus (higher means more bonus for failure/SLA violation)
        self.decay_rate_factor = decay_rate_factor # Controls how fast older memories decay

    def log_episode(self, episode_data: Dict[str, Any]):
        self.episodic_logs.append(episode_data)

    def distill_strategy(self, outcome_data: Dict[str, Any]):
        agreed_config = outcome_data.get("agreed_config")
        final_metrics = outcome_data.get("final_metrics")
        sla_violation_occurred = outcome_data.get("sla_violation_occurred")
        saved_energy_percent = outcome_data.get("saved_energy_percent")
        unresolved_negotiation = outcome_data.get("unresolved_negotiation")
        
        current_traffic_bps = final_metrics.get("current_traffic_arrival_rate_bps", 0)
        if current_traffic_bps < network_simulator._TRAFFIC_BASE_SLICE * 0.8: # Use network_simulator instance
            traffic_level = "low"
        elif current_traffic_bps > network_simulator._TRAFFIC_BASE_SLICE * 1.2: # Use network_simulator instance
            traffic_level = "high"
        else:
            traffic_level = "medium"

        # Get trial_number from outcome_data
        trial_number = outcome_data.get("trial_number")

        # Initialize action_data and outcome_summary_data to ensure consistent structure
        action_data = {
            "last_ran_proposal_mhz": None, # Renamed for clarity
            "last_edge_proposal_ghz": None # Renamed for clarity
        }
        outcome_summary_data = {}

        # Define strategy type based on outcome
        if unresolved_negotiation:
            event_type = "failed_negotiation_strategy" # Negotiation itself failed
            negotiation_result_summary = "unresolved_negotiation"
            description_prefix = "Negotiation failed to reach agreement"
            
            # Safely get last proposals, which might be None
            last_ran_prop_dict = outcome_data.get("last_ran_proposal")
            last_edge_prop_dict = outcome_data.get("last_edge_proposal")

            # Safely extract values
            action_data["last_ran_proposal_mhz"] = last_ran_prop_dict.get("ran_bandwidth_mhz") if last_ran_prop_dict else None
            action_data["last_edge_proposal_ghz"] = last_edge_prop_dict.get("edge_cpu_frequency_ghz") if last_edge_prop_dict else None


            outcome_summary_data = {
                "negotiation_result": negotiation_result_summary,
                "reason_for_failure": "Max iterations reached or agent declared no agreement possible.",
                "latency_ms_at_failure": final_metrics.get("latency_ms"),
                "energy_consumption_watts_at_failure": final_metrics.get("energy_consumption_watts"),
                "sla_violation_occurred_at_failure": bool(sla_violation_occurred) # Capture SLA violation even if unresolved
            }
            description_suffix = f"Last known latency was {final_metrics.get('latency_ms'):.2f}ms (SLA {'violated' if sla_violation_occurred else 'met'}). Last energy consumption was {final_metrics.get('energy_consumption_watts'):.2f}W."

        elif agreed_config and sla_violation_occurred:
            event_type = "failed_agreement_strategy" # Agreement reached, but led to SLA violation
            negotiation_result_summary = "agreement_with_sla_violation"
            description_prefix = "Agreement reached but led to SLA violation"
            
            # For agreed configs that violate SLA, the agreed config IS the "last proposal" that failed
            action_data["last_ran_proposal_mhz"] = agreed_config["ran_bw"]
            action_data["last_edge_proposal_ghz"] = agreed_config["edge_cpu"]

            outcome_summary_data = {
                "negotiation_result": negotiation_result_summary,
                "latency_ms": final_metrics.get("latency_ms"),
                "energy_consumption_watts": final_metrics.get("energy_consumption_watts"),
                "saved_energy_percent": saved_energy_percent,
                "sla_violation_occurred": bool(sla_violation_occurred),
                "reason_for_failure": "Agreement led to SLA violation." # Explicitly add reason
            }
            # Corrected typo: agged_config -> agreed_config
            description_suffix = f"with BW {agreed_config['ran_bw']:.1f} MHz and CPU {agreed_config['edge_cpu']:.1f} GHz under {traffic_level} traffic. Latency was VIOLATED ({final_metrics.get('latency_ms'):.2f}ms). Energy savings: {saved_energy_percent:.2f}%."

        elif agreed_config and not sla_violation_occurred:
            event_type = "successful_agreement_strategy" # Successful agreement without SLA violation
            negotiation_result_summary = "success"
            description_prefix = "Successfully achieved agreement"
            
            # For successful agreements, the agreed config is also the "last proposal" that succeeded
            action_data["last_ran_proposal_mhz"] = agreed_config["ran_bw"]
            action_data["last_edge_proposal_ghz"] = agreed_config["edge_cpu"]

            outcome_summary_data = {
                "negotiation_result": negotiation_result_summary,
                "latency_ms": final_metrics.get("latency_ms"),
                "energy_consumption_watts": final_metrics.get("energy_consumption_watts"),
                "saved_energy_percent": saved_energy_percent,
                "sla_violation_occurred": bool(sla_violation_occurred)
            }
            description_suffix = f"with BW {agreed_config['ran_bw']:.1f} MHz and CPU {agreed_config['edge_cpu']:.1f} GHz under {traffic_level} traffic. Latency was met. Energy savings: {saved_energy_percent:.2f}%."
        else:
            # Should not happen if logic is sound, but as a fallback
            print(f"Warning: Unknown outcome type for distillation: {outcome_data}")
            return

        strategy = {
            "event_type": event_type,
            "context": {
                "traffic_level_category": traffic_level, # Keep for semantic context
                "current_traffic_arrival_rate_bps": current_traffic_bps, # Exact value
                "sla_latency_threshold_ms": SLA_LATENCY_THRESHOLD_MS,
                "time_step": final_metrics.get("current_time_step"),
                "trial_number": trial_number,
                "ran_bw_range": f"{e2_api_tool.MIN_RAN_BW_MHZ}-{e2_api_tool.MAX_RAN_BW_MHZ}",
                "edge_cpu_range": f"0-{e2_api_tool.MAX_EDGE_CPU_GHZ}"
            },
            "action": action_data, # Contains proposed/agreed BW and CPU
            "outcome_summary": outcome_summary_data, # Contains latency, energy, SLA status
            "description": f"{description_prefix} under {traffic_level} traffic. {description_suffix}"
        }
        self.distilled_strategies.append(strategy)

    def _tokenize_text(self, text: str) -> set:
        return set(re.findall(r'\b\w+\b', text.lower()))

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def query_memory(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[CollectiveMemory] Query context received by query_memory: {query_context}")

        current_trial_number = query_context.get("current_trial_number", 0) 
        agent_role = query_context.get("agent_role", "").lower()
        query_keywords_set = self._tokenize_text(" ".join(query_context.get("keywords", [])))
        
        # Step 1: Calculate base scores for all strategies
        scored_candidates = []
        for strategy in self.distilled_strategies:
            # Semantic Similarity (Jaccard)
            strategy_text = strategy["description"] + " " + json.dumps(strategy["outcome_summary"])
            strategy_keywords_set = self._tokenize_text(strategy_text)
            semantic_similarity = self._jaccard_similarity(query_keywords_set, strategy_keywords_set)

            # Time-Decay Score
            age = current_trial_number - strategy["context"].get("trial_number", 0)
            # Using exp(-age / decay_rate_factor) for continuous decay.
            time_decay_score = np.exp(-max(0, age) / self.decay_rate_factor) 
            
            # Combine to get initial score (always include semantic and time-decay)
            base_score = (self.alpha * semantic_similarity) + (self.beta * time_decay_score)
            
            # Apply Inflection Bonus ONLY if debiasing is enabled
            if self.debiasing_enabled:
                inflection_bonus = 0.0
                if strategy["outcome_summary"].get("negotiation_result") in ["unresolved_negotiation", "agreement_with_sla_violation"]:
                    if strategy["outcome_summary"].get("sla_violation_occurred_at_failure") or \
                       strategy["outcome_summary"].get("sla_violation_occurred"):
                        inflection_bonus = 1.0 # High bonus for direct SLA violation failures
                    else:
                        inflection_bonus = 0.5 # Lower bonus for other negotiation failures
                base_score += (self.delta * inflection_bonus)
            
            scored_candidates.append({"strategy": strategy, "base_score": base_score, "semantic_keywords": strategy_keywords_set})
        
        # Sort by base score descending
        scored_candidates.sort(key=lambda x: x["base_score"], reverse=True)

        selected_strategies = []
        top_n = 5 # Number of top strategies to retrieve

        if self.debiasing_enabled:
            # Step 2: Iterative Selection with Diversity Penalty (ETR-DDS) - ONLY if debiasing is enabled
            selected_keywords_union = set() # Union of keywords of selected strategies for diversity calculation

            for _ in range(top_n):
                best_candidate = None
                max_final_score = -np.inf

                for candidate in scored_candidates:
                    if candidate["strategy"] in selected_strategies: # Skip already selected
                        continue

                    diversity_penalty = 0.0
                    if selected_strategies: # Only apply penalty if some strategies are already selected
                        # Calculate similarity of current candidate to the *set* of already selected strategies
                        similarity_to_selected = self._jaccard_similarity(candidate["semantic_keywords"], selected_keywords_union)
                        diversity_penalty = self.gamma * similarity_to_selected
                    
                    final_score = candidate["base_score"] - diversity_penalty

                    if final_score > max_final_score:
                        max_final_score = final_score
                        best_candidate = candidate
                
                if best_candidate:
                    selected_strategies.append(best_candidate["strategy"])
                    selected_keywords_union.update(best_candidate["semantic_keywords"])
                    # Log age of queried strategies
                    age = current_trial_number - best_candidate["strategy"]["context"].get("trial_number", 0)
                    global all_queried_strategies_ages
                    all_queried_strategies_ages.append(age)
                else:
                    break # No more candidates to select
        else:
            # Vanilla memory: Just take the top N based on base_score (semantic + time-decay)
            selected_strategies = [candidate["strategy"] for candidate in scored_candidates[:top_n]]
            for strategy in selected_strategies:
                age = current_trial_number - strategy["context"].get("trial_number", 0)
                all_queried_strategies_ages.append(age)


        # Update global counters for queried strategies
        global total_successful_strategies_queried
        global total_failed_strategies_queried
        for strategy in selected_strategies:
            if strategy["outcome_summary"].get("negotiation_result") == "success":
                total_successful_strategies_queried += 1
            elif strategy["outcome_summary"].get("negotiation_result") in ["unresolved_negotiation", "agreement_with_sla_violation"]:
                total_failed_strategies_queried += 1

        # Infer/distill best strategy - make it bolder and traffic-aware
        inferred_successful_config = None
        successful_configs = []
        for s in selected_strategies:
            if s["outcome_summary"].get("negotiation_result") == "success":
                ran_bw = s["action"].get("last_ran_proposal_mhz")
                edge_cpu = s["action"].get("last_edge_proposal_ghz")
                if ran_bw is not None and edge_cpu is not None:
                    successful_configs.append({"ran_bandwidth_mhz": ran_bw, "edge_cpu_frequency_ghz": edge_cpu})

        if successful_configs:
            avg_ran_bw = sum([c["ran_bandwidth_mhz"] for c in successful_configs]) / len(successful_configs)
            avg_edge_cpu = sum([c["edge_cpu_frequency_ghz"] for c in successful_configs]) / len(successful_configs)
            
            bolder_ran_bw = avg_ran_bw
            bolder_edge_cpu = avg_edge_cpu

            traffic_level_category = query_context.get("traffic_level_category")
            
            if agent_role == "ran":
                current_latency = query_context.get("latency_ms", 0)
                if current_latency > SLA_LATENCY_THRESHOLD_MS * 0.9:
                    bolder_ran_bw = min(e2_api_tool.MAX_RAN_BW_MHZ, round(avg_ran_bw * 1.15, 1))
                elif traffic_level_category == "high":
                    bolder_ran_bw = min(e2_api_tool.MAX_RAN_BW_MHZ, round(avg_ran_bw * 1.05, 1))
                else:
                    bolder_ran_bw = max(e2_api_tool.MIN_RAN_BW_MHZ, round(avg_ran_bw * 0.85, 1))

            elif agent_role == "edge":
                if traffic_level_category == "high":
                    bolder_edge_cpu = min(e2_api_tool.MAX_EDGE_CPU_GHZ, round(avg_edge_cpu * 1.15, 1))
                else:
                    bolder_edge_cpu = round(avg_edge_cpu * 1.05, 1)

            inferred_successful_config = {
                "ran_bandwidth_mhz": bolder_ran_bw,
                "edge_cpu_frequency_ghz": bolder_edge_cpu
            }
        
        patterns_to_avoid = []
        for s in selected_strategies:
            if s["outcome_summary"].get("negotiation_result") in ["unresolved_negotiation", "agreement_with_sla_violation"]:
                patterns_to_avoid.append({
                    "last_ran_proposal_mhz": s["action"].get("last_ran_proposal_mhz"),
                    "last_edge_proposal_ghz": s["action"].get("last_edge_proposal_ghz"),
                    "reason": s["outcome_summary"].get("reason_for_failure"),
                    "sla_violation_occurred": s["outcome_summary"].get("sla_violation_occurred_at_failure") or \
                                              s["outcome_summary"].get("sla_violation_occurred"),
                    "context": s["context"]
                })

        return {
            "retrieved_strategies": selected_strategies,
            "inferred_successful_config": inferred_successful_config,
            "patterns_to_avoid": patterns_to_avoid
        }


    def reset_memory(self):
        self.episodic_logs = []
        self.distilled_strategies = []
