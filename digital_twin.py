import numpy as np
from typing import Dict, Any
from network_simulator import NetworkSimulator
from config import NUM_TIME_STEPS

class DigitalTwin(NetworkSimulator):
    """
    A digital copy of the NetworkSimulator. Agents use this to test the
    outcome of proposed actions without affecting the live environment.
    """
    def __init__(self, simulator_params: Dict[str, Any]):
        super().__init__()
        # Override parameters to match the main simulator's current configuration
        self.fmax = simulator_params.get('fmax', 45.0)
        self.tau = simulator_params.get('tau', 0.01)
        self.U = simulator_params.get('U', 0.0017)
        self.min_spectral_efficiency = simulator_params.get('min_spectral_efficiency', 6.0)
        self.max_spectral_efficiency = simulator_params.get('max_spectral_efficiency', 8.0)
        
        # Placeholder traffic data; set to a single-element array to represent the "current" context
        self.traffic = np.array([[0]])  
        self.arrival_rate = 0.0

    def reset_to_current_state(self, current_metrics: Dict[str, Any]):
        """Resets the Digital Twin's state to match the main simulator's current metrics."""
        self.cqueue = current_metrics["cqueue_bits"]
        self.rqueue = current_metrics["rqueue_bits"]
        self.Q = current_metrics["cqueue_bits"] + current_metrics["rqueue_bits"]
        self.latency = current_metrics["latency_ms"] / 1000.0
        self.transmission_rate = current_metrics["transmission_rate_bps"]
        self.energy_consumption_watts = current_metrics["energy_consumption_watts"]
        self.cpu_frequency_allocated = current_metrics["cpu_frequency_ghz_allocated"]
        self.bandwidth_allocated = current_metrics["bandwidth_mhz_allocated"]
        self.cpu_allocation_conflict_count_internal = current_metrics["cpu_allocation_conflict_count"]
        
        # In a dataset-driven environment, we use the absolute time step
        self.t = current_metrics["current_time_step"]
        # id_traffic is safe here because we store the specific arrival rate below
        self.id_traffic = 0 
        
        self.current_spectral_efficiency = current_metrics["current_spectral_efficiency_bits_per_hz_per_s"]
        
        # Store the current traffic arrival rate as the only element in our DT's traffic array
        self.traffic = np.array([[current_metrics["current_traffic_arrival_rate_bps"]]])
        self.arrival_rate = current_metrics["average_traffic_arrival_rate_bps"]

    def get_metrics(self, **kwargs) -> Dict[str, Any]:
        """
        Returns the current state of the Digital Twin.
        Updated to accept **kwargs to maintain interface alignment with the live Simulator/E2 API.
        """
        # The DT's traffic array has exactly one element for the current prediction context.
        current_traffic = float(self.traffic[0, 0]) if self.traffic.size > 0 else 0.0

        return {
            "latency_ms": self.latency * 1000,
            "transmission_rate_bps": self.transmission_rate,
            "cqueue_bits": self.cqueue,
            "rqueue_bits": self.rqueue,
            "energy_consumption_watts": self.energy_consumption_watts,
            "cpu_frequency_ghz_allocated": self.cpu_frequency_allocated,
            "bandwidth_mhz_allocated": self.bandwidth_allocated,
            "cpu_allocation_conflict_count": self.cpu_allocation_conflict_count_internal,
            "current_time_step": self.t,
            "current_traffic_arrival_rate_bps": current_traffic,
            "average_traffic_arrival_rate_bps": self.arrival_rate,
            "current_spectral_efficiency_bits_per_hz_per_s": self.current_spectral_efficiency
        }

    def simulate_step_for_prediction(self, proposed_ran_bandwidth_mhz: float, proposed_edge_cpu_frequency_ghz: float) -> Dict[str, float]:
        """
        Simulates one step forward with proposed actions to predict outcomes.
        Does not alter the main simulator's state.
        """
        # Store current state to restore after prediction logic
        original_state = self.get_metrics()
        original_cqueue = self.cqueue
        original_rqueue = self.rqueue
        
        # Temporarily apply proposed actions
        self.bandwidth_allocated = proposed_ran_bandwidth_mhz
        self.cpu_frequency_allocated = proposed_edge_cpu_frequency_ghz

        # Perform one simulation step based on the traffic context stored in reset_to_current_state
        current_traffic_at_id_dt = self.traffic[0, 0]
        arriving_data_in_tau_dt = self.tau * current_traffic_at_id_dt
        U_proc_capacity_in_tau_dt = self.tau * self.U * self.cpu_frequency_allocated * (10**9)

        # 1. Update Computation Queue
        cqueue_dt = original_cqueue + arriving_data_in_tau_dt
        processed_data_in_tau_dt = min(cqueue_dt, U_proc_capacity_in_tau_dt)
        cqueue_dt = max(0, cqueue_dt - processed_data_in_tau_dt)

        # 2. Update Radio Queue
        rqueue_dt = original_rqueue + processed_data_in_tau_dt
        radio_capacity_in_tau_dt = self.tau * self._calculate_transmission_rate(self.bandwidth_allocated)
        transmitted_data_in_tau_dt = min(rqueue_dt, radio_capacity_in_tau_dt)
        rqueue_dt = max(0, rqueue_dt - transmitted_data_in_tau_dt)

        # 3. Calculate Predicted Metrics
        total_Q_dt = cqueue_dt + rqueue_dt
        predicted_latency_sec_dt = 0.0
        
        if total_Q_dt > 0 and transmitted_data_in_tau_dt > 0:
            predicted_latency_sec_dt = total_Q_dt / (transmitted_data_in_tau_dt / self.tau)
        elif total_Q_dt > 0:
            predicted_latency_sec_dt = 1.0 # Represent a stalled system

        predicted_latency_ms_dt = predicted_latency_sec_dt * 1000
        predicted_energy_watts_dt = self._calculate_energy_consumption(self.bandwidth_allocated, self.cpu_frequency_allocated)

        # 4. Conflict Checking
        cpu_conflict_dt = original_state["cpu_allocation_conflict_count"]
        if proposed_edge_cpu_frequency_ghz > self.fmax:
            cpu_conflict_dt += 1
            predicted_latency_ms_dt = max(predicted_latency_ms_dt, 500.0) # Penalty for conflict

        # Restore the DT to its original state so it can be reused for other proposals
        self.reset_to_current_state(original_state)

        return {
            "predicted_latency_ms": predicted_latency_ms_dt,
            "predicted_energy_watts": predicted_energy_watts_dt,
            "predicted_cpu_allocation_conflict_count": cpu_conflict_dt
        }