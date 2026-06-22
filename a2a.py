import re
import json
from typing import Dict, Any, Optional

from agents import RanAgent, EdgeAgent
from e2_api_tool import E2APISimulator
from collective_memory import CollectiveMemory
from config import SLA_LATENCY_THRESHOLD_MS, REFERENCE_ENERGY_FOR_SAVINGS_W


class A2ANegotiationManager:
    """
    Manages the iterative negotiation process between the RAN Agent and the Edge Agent.
    Synchronizes state between the live simulator, the agents' internal Digital Twins, 
    and the Collective Memory.
    """
    def __init__(self, ran_agent: RanAgent, edge_agent: EdgeAgent, e2_api: E2APISimulator, 
                 collective_memory: Optional[CollectiveMemory], max_iterations: int = 8, trial_num: int = 0):
        self.ran_agent = ran_agent
        self.edge_agent = edge_agent
        self.e2_api = e2_api
        self.collective_memory = collective_memory
        self.max_iterations = max_iterations
        self.negotiation_log = []
        self.agreed_config = {"ran_bw": None, "edge_cpu": None}
        self.last_ran_proposal = None
        self.last_edge_proposal = None
        self.consensus_time = -1
        self.unresolved_negotiation = False
        self.unparseable_message_failure = False 
        self.negotiation_status = "ongoing"
        self.trial_num = trial_num 

    @staticmethod
    def _parse_agent_message(message: str) -> Dict[str, Any]:
        """
        Parses the string response from an LLM agent to extract the intent and parameters.
        Matches the expected PROPOSE_ACTION, ACCEPT_AGREEMENT, and NO_AGREEMENT_POSSIBLE formats.
        """
        message = message.strip()
        
        propose_match = re.search(r"PROPOSE_ACTION:\s*(\{.*?\})", message, re.IGNORECASE | re.DOTALL)
        accept_match = re.search(r"ACCEPT_AGREEMENT:\s*(\{.*?\})", message, re.IGNORECASE | re.DOTALL)
        no_agreement_match = re.search(r"NO_AGREEMENT_POSSIBLE", message, re.IGNORECASE)

        if propose_match:
            intent = "PROPOSE_ACTION"
            params_str = propose_match.group(1)
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse PROPOSE_ACTION JSON: {params_str}. Error: {e}")
                params = {}
            params.setdefault("ran_bandwidth_mhz", None)
            params.setdefault("edge_cpu_frequency_ghz", None)
            return {"intent": intent, "parameters": params}
        elif accept_match:
            intent = "ACCEPT_AGREEMENT"
            params_str = accept_match.group(1)
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse ACCEPT_AGREEMENT JSON: {params_str}. Error: {e}")
                params = {}
            params.setdefault("ran_bandwidth_mhz", None)
            params.setdefault("edge_cpu_frequency_ghz", None)
            return {"intent": intent, "parameters": params}
        elif no_agreement_match:
            return {"intent": "NO_AGREEMENT_POSSIBLE", "parameters": {}}
        else:
            return {"intent": "PARSING_FAILED", "parameters": {"reason": "Message format invalid"}}

    def run_negotiation(self) -> Dict[str, Any]:
        """
        Executes the full negotiation loop.
        Updates metrics from the live simulator (E2 API) and coordinates agent moves.
        """
        self.consensus_time = -1
        self.unresolved_negotiation = False
        self.unparseable_message_failure = False 
        self.agreed_config = {"ran_bw": None, "edge_cpu": None}
        self.negotiation_log = []
        self.last_ran_proposal = None
        self.last_edge_proposal = None
        self.negotiation_status = "ongoing"
        
        print("\n--- Starting A2A Negotiation ---")
        # Initial state observation from the live simulator
        current_metrics = self.e2_api.get_metrics()
        
        ran_message_to_start = "Hello Edge Agent, I'm the RAN Agent. My goal is to optimize energy efficiency by reducing bandwidth while ensuring good performance. Let's find a good balance. What are your initial proposals?"
        edge_message_to_start = "Hello RAN Agent, I'm the Edge Agent. My goal is to minimize latency for the cross-domain slice. I'm ready to find optimal values."

        print(f"\n[{self.ran_agent.name}] Initializing...")
        print(f"[{self.edge_agent.name}] Initializing...")

        # RAN AGENT Starts the negotiation
        ran_response_text = self.ran_agent.make_negotiation_move(
            opposing_agent_message=edge_message_to_start,
            current_metrics=current_metrics,
            iteration=0,
            max_iterations=self.max_iterations,
            negotiation_ended=False,
            current_trial_num=self.trial_num
        )
        
        parsed_ran_move = self._parse_agent_message(ran_response_text)
        self.negotiation_log.append({
            "iteration": 0,
            "agent": self.ran_agent.name,
            "message": ran_response_text,
            "parsed": parsed_ran_move,
            "metrics_before_move": current_metrics
        })

        if parsed_ran_move["intent"] == "PROPOSE_ACTION":
            self.last_ran_proposal = parsed_ran_move["parameters"]
        elif parsed_ran_move["intent"] == "PARSING_FAILED":
            print(f"RAN agent's initial message could not be parsed. Ending negotiation.")
            self.unresolved_negotiation = True
            self.unparseable_message_failure = True
            self.negotiation_status = "unresolved"
            return self._finalize_negotiation_results(current_metrics)

        # Main Negotiation Loop
        for i in range(1, self.max_iterations):
            if self.negotiation_status != "ongoing":
                break

            print(f"\n--- Negotiation Round {i+1}/{self.max_iterations} ---")

            # 1. EDGE AGENT'S TURN
            edge_response_text = self.edge_agent.make_negotiation_move(
                opposing_agent_message=ran_response_text,
                current_metrics=current_metrics,
                iteration=i,
                max_iterations=self.max_iterations,
                negotiation_ended=False,
                current_trial_num=self.trial_num
            )
            
            parsed_edge_move = self._parse_agent_message(edge_response_text)
            self.negotiation_log.append({
                "iteration": i,
                "agent": self.edge_agent.name,
                "message": edge_response_text,
                "parsed": parsed_edge_move,
                "metrics_before_move": current_metrics
            })
            
            # Refresh live metrics for the next participant
            current_metrics = self.e2_api.get_metrics()

            if parsed_edge_move["intent"] == "PROPOSE_ACTION":
                self.last_edge_proposal = parsed_edge_move["parameters"]
            elif parsed_edge_move["intent"] == "ACCEPT_AGREEMENT":
                ran_bw = parsed_edge_move["parameters"].get("ran_bandwidth_mhz")
                edge_cpu = parsed_edge_move["parameters"].get("edge_cpu_frequency_ghz")
                
                # Verify that what is being accepted matches the last valid proposal
                if self.last_ran_proposal and \
                   ran_bw == self.last_ran_proposal.get("ran_bandwidth_mhz") and \
                   edge_cpu == self.last_ran_proposal.get("edge_cpu_frequency_ghz"):
                    
                    self.agreed_config = {"ran_bw": ran_bw, "edge_cpu": edge_cpu}
                    enforcement = self.e2_api.enforce_actions(ran_bw, edge_cpu)
                    if enforcement["status"] == "success":
                        self.consensus_time = i + 1
                        self.negotiation_status = "agreed"
                        break
            
            elif parsed_edge_move["intent"] in ["NO_AGREEMENT_POSSIBLE", "PARSING_FAILED"]:
                self.unresolved_negotiation = True
                self.negotiation_status = "unresolved"
                if parsed_edge_move["intent"] == "PARSING_FAILED":
                    self.unparseable_message_failure = True
                break

            # 2. RAN AGENT'S TURN
            ran_response_text = self.ran_agent.make_negotiation_move(
                opposing_agent_message=edge_response_text,
                current_metrics=current_metrics,
                iteration=i,
                max_iterations=self.max_iterations,
                negotiation_ended=False,
                current_trial_num=self.trial_num
            )
            
            parsed_ran_move = self._parse_agent_message(ran_response_text)
            self.negotiation_log.append({
                "iteration": i,
                "agent": self.ran_agent.name,
                "message": ran_response_text,
                "parsed": parsed_ran_move,
                "metrics_before_move": current_metrics
            })

            current_metrics = self.e2_api.get_metrics()

            if parsed_ran_move["intent"] == "PROPOSE_ACTION":
                self.last_ran_proposal = parsed_ran_move["parameters"]
            elif parsed_ran_move["intent"] == "ACCEPT_AGREEMENT":
                ran_bw = parsed_ran_move["parameters"].get("ran_bandwidth_mhz")
                edge_cpu = parsed_ran_move["parameters"].get("edge_cpu_frequency_ghz")

                if self.last_edge_proposal and \
                   ran_bw == self.last_edge_proposal.get("ran_bandwidth_mhz") and \
                   edge_cpu == self.last_edge_proposal.get("edge_cpu_frequency_ghz"):

                    self.agreed_config = {"ran_bw": ran_bw, "edge_cpu": edge_cpu}
                    enforcement = self.e2_api.enforce_actions(ran_bw, edge_cpu)
                    if enforcement["status"] == "success":
                        self.consensus_time = i + 1
                        self.negotiation_status = "agreed"
                        break
            
            elif parsed_ran_move["intent"] in ["NO_AGREEMENT_POSSIBLE", "PARSING_FAILED"]:
                self.unresolved_negotiation = True
                self.negotiation_status = "unresolved"
                if parsed_ran_move["intent"] == "PARSING_FAILED":
                    self.unparseable_message_failure = True
                break

        return self._finalize_negotiation_results(current_metrics)

    def _finalize_negotiation_results(self, last_known_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates final KPIs and saves the outcome to Collective Memory.
        Handles the scenario where negotiation ends without agreement.
        """
        final_metrics = self.e2_api.get_metrics()
        
        # If negotiation was unresolved, use the state at the point of failure
        if self.negotiation_status != "agreed":
            self.consensus_time = self.max_iterations + 1
            self.unresolved_negotiation = True
            final_metrics = last_known_metrics

        final_energy = final_metrics.get("energy_consumption_watts", 0.0)
        final_latency = final_metrics.get("latency_ms", 0.0)
        
        # Calculate energy savings against the reference baseline
        saved_energy_percent = 0.0
        if self.negotiation_status == "agreed":
            saved_energy_percent = ((REFERENCE_ENERGY_FOR_SAVINGS_W - final_energy) / REFERENCE_ENERGY_FOR_SAVINGS_W) * 100 \
                if REFERENCE_ENERGY_FOR_SAVINGS_W != 0 else 0.0

        sla_violation_occurred = final_latency > SLA_LATENCY_THRESHOLD_MS

        # Distill the experience into the Collective Memory for future trials
        if self.collective_memory:
            self.collective_memory.distill_strategy({
                "agreed_config": self.agreed_config,
                "final_metrics": final_metrics,
                "sla_violation_occurred": sla_violation_occurred,
                "saved_energy_percent": saved_energy_percent,
                "unresolved_negotiation": self.unresolved_negotiation,
                "last_ran_proposal": self.last_ran_proposal,
                "last_edge_proposal": self.last_edge_proposal,
                "trial_number": self.trial_num
            })

        print(f"\n--- Negotiation Ended (Status: {self.negotiation_status}) ---")
        
        return {
            "agreed_config": self.agreed_config,
            "negotiation_log": self.negotiation_log,
            "consensus_time": self.consensus_time,
            "unresolved_negotiation": self.unresolved_negotiation,
            "unparseable_message_failure": self.unparseable_message_failure,
            "simulator_internal_cpu_conflicts": final_metrics.get("cpu_allocation_conflict_count", 0),
            "saved_energy_percent": saved_energy_percent,
            "sla_violation_occurred": sla_violation_occurred,
            "average_energy_this_trial": final_energy,
            "average_latency_this_trial": final_latency
        }