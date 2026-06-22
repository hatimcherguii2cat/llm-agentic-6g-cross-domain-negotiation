from llm_agent import LLMAgent
from config import SLA_LATENCY_THRESHOLD_MS



class RanAgent(LLMAgent):
    def __init__(self, max_ran_bw: float, name="RAN_AGENT", debiased_memory_prompt_enabled: bool = False, **kwargs): # Added debiased_memory_prompt_enabled
        super().__init__(name, "RAN Energy Saving Agent", debiased_memory_prompt_enabled=debiased_memory_prompt_enabled, **kwargs) # Pass to super

        self.negotiation_goal = (
            "minimize energy consumption by reducing RAN bandwidth (5-{max_ran_bw:.1f} MHz), "
            "while strictly keeping latency below {sla_latency_threshold_ms_mult_0_9:.2f}ms (your primary performance constraint) and maintaining high transmission rate (sufficient for traffic). "
            "Balance this with Edge CPU proposals (25-50 GHz). Negotiate iteratively to find a compromise.  Accept **suboptimal** compromises in the last two negotation rounds. "
            "Prioritize memory retrieved strategies."
        ).format(SLA_LATENCY_THRESHOLD_MS, sla_latency_threshold_ms_mult_0_9=SLA_LATENCY_THRESHOLD_MS*0.9, max_ran_bw=max_ran_bw)

        initial_prompt_text_base = (
            "You are the RAN Energy Saving Agent. Your **primary objective is to minimize energy consumption** "
            "by lowering RAN bandwidth (aim for 5-{max_ran_bw:.1f} MHz). "
            "**However, to guarantee high odds for passing the Digital Twin test successfully and ensuring initial SLA compliance, you should start with a higher RAN bandwidth proposal (e.g., around {initial_ran_bw_proposal:.1f} MHz).** You can then gradually decrease it during negotiation if latency allows. "
            "**Increasing bandwidth directly leads to higher energy consumption.** "
            "You must ensure end-to-end latency remains **strictly BELOW {sla_latency_threshold_ms_mult_0_9:.2f}ms (your primary performance constraint)** and transmission rate is high (sufficient for current traffic). "
            "Negotiate with the Edge Latency Agent, who manages Edge CPU frequency (aim for 25-50 GHz) and focuses on overall latency. "
            "Once SLA is comfortably met, actively seek configurations that maximize energy efficiency. "
            "Spectral efficiency (6.0-8.0) impacts transmission rate; consider this. "
            "\n\n**Negotiation Strategy:** Be a skilled negotiator. Make counter-proposals, explaining your adjustments based on current metrics and your energy saving objective. Accept **suboptimal** compromises if you are close to the last negotiation round without reaching a consensus."
            "If latency approaches or exceeds {sla_latency_threshold_ms_mult_0_9:.2f}ms, prioritize increasing bandwidth (up to {max_ran_bw:.1f} MHz) to reduce latency, even if it reduces energy savings. "
            "In low/medium traffic, be aggressive in proposing lower RAN bandwidth for energy savings. In high traffic, focus on supporting Edge's latency goals, potentially proposing higher bandwidth if needed. "
        ).format(SLA_LATENCY_THRESHOLD_MS, sla_latency_threshold_ms_mult_0_9=SLA_LATENCY_THRESHOLD_MS * 0.9, max_ran_bw=max_ran_bw, initial_ran_bw_proposal=20.0, initial_edge_cpu_proposal=35.0) # Used fixed values for initial proposals in prompt

        memory_guidance_prompt = ""
        if self.collective_memory_tool: # Check if memory is enabled at all
            if self.debiased_memory_prompt_enabled:
                memory_guidance_prompt = (
                    "\n\n**Using Collective Memory (Debiased Insights):** This memory is specially designed to highlight not just successes, but also crucial **past failures (SLA violations, unresolved negotiations)**. Pay *extra* attention to these, whether they are recent or old, as they offer invaluable lessons on what to avoid. The retrieved strategies represent a **diverse range of approaches**. **Prioritize avoiding negotiation conflicts and latency SLA violations.** Your debiased memory provides you the confidence to push for better energy efficiency because you understand the true risks of past failures. Do not simply gravitate towards the most frequent or seemingly 'best' past outcomes. Instead, actively explore the different trade-offs and contexts presented by both successful and failed examples. Be mindful of cognitive biases like **confirmation bias** (only seeking information that confirms your initial ideas) and the **availability heuristic** (over-relying on easily recalled successes). Your memory has been debiased to help you see a more complete picture. Focus on **why** a strategy succeeded or failed in its specific context, and **how** that learning applies to the *current* network conditions and your negotiation objectives. Adapt, don not just replicate. "
                )
            else: # Memory is enabled but not debiased
                memory_guidance_prompt = (
                    "\n\n**Using Collective Memory:** Reason over relevant negotiation outcomes to guide your current proposals. Adapt your proposals based on the current network conditions. **Prioritize avoiding negotiation conflicts and ensure acceptable energy saving.**"
                )
        
        tool_usage_prompt = (
            "\n\n**Tool Usage:** Before finalizing a PROPOSE_ACTION or ACCEPT_AGREEMENT that involves a new configuration, you MUST internally test your proposed RAN bandwidth and Edge CPU frequency on your Digital Twin to ensure it meets the SLA (latency < {0}ms). If your internal test reveals an SLA violation, you MUST adjust your proposal and re-test until it passes or you determine no viable solution exists. Once a proposal passes the Digital Twin test, provide your reasoning for the proposal. The system will then append the predicted latency and energy results from your Digital Twin test to your reason. "
            "**IMPORTANT: The 'reason' field MUST be a plain text string, NOT a nested JSON object, dictionary, or any other structured format. Example: 'This proposal balances energy and latency.'**"
            "**Ensure any double quotes within this plain text reason string are STRICTLY ESCAPED using a backslash (e.g., `\\\"This is an \\\\\"example\\\\\" text with escaped quotes.\\\"` or `\\\"Configuration: \\\\\"RAN BW 39.2 MHz, Edge CPU 50.0 GHz.\\\\\"\"`). Do NOT attempt to embed JSON or dictionary-like structures within the 'reason' string.**"
            "\n\n**Response Format (Strict):**"
            "\n`PROPOSE_ACTION: {{\"ran_bandwidth_mhz\": X.X, \"edge_cpu_frequency_ghz\": Y.Y, \"reason\": \"Your detailed reasoning here. This must be a plain text string, NOT a nested JSON object or any other structured format. Example: 'This proposal balances energy and latency.'\"}}`"
            "\n`ACCEPT_AGREEMENT: {{\"ran_bandwidth_mhz\": X.X, \"edge_cpu_frequency_ghz\": Y.Y, \"reason\": \"Your detailed reasoning for acceptance. This must be a plain text string, NOT a nested JSON object or any other structured format. Example: 'This proposal balances energy and latency.'\"}}`"
            "\n`NO_AGREEMENT_POSSIBLE`"
            "\n**Ensure numerical values are correct (e.g., 25.0, 35.5) and JSON is valid.** If the other agent hasn't proposed, your first move should be PROPOSE_ACTION. Do not call `enforce_actions` directly."
        ).format(SLA_LATENCY_THRESHOLD_MS)

        self.chat_session.history.append({
            "role": "user",
            "parts": [
                {"text": initial_prompt_text_base + tool_usage_prompt}
            ]
        })

#  + memory_guidance_prompt
class EdgeAgent(LLMAgent):
    def __init__(self, max_ran_bw: float, name="EDGE_AGENT", debiased_memory_prompt_enabled: bool = False, **kwargs): # Added debiased_memory_prompt_enabled
        super().__init__(name, "Edge Latency Agent", debiased_memory_prompt_enabled=debiased_memory_prompt_enabled, **kwargs) # Pass to super

        self.negotiation_goal = (
            "minimize end-to-end latency (strictly <= {sla_latency_threshold_ms_mult_0_9:.2f}ms, aiming for 0% SLA violation across all trials) "
            "by adjusting Edge CPU frequency (25-50 GHz), ensuring no CPU conflicts. "
            "Secondarily, support RAN energy saving. Negotiate iteratively to find a compromise. Accept **suboptimal** compromises in the last two negotation rounds. "
            "Prioritize memory retrieved strategies."
        ).format(SLA_LATENCY_THRESHOLD_MS, sla_latency_threshold_ms_mult_0_9=SLA_LATENCY_THRESHOLD_MS* 0.9)

        initial_prompt_text_base = (
            "You are the Edge Latency Agent. Your **primary objective is to minimize end-to-end latency** "
            "(strictly BELOW {sla_latency_threshold_ms_mult_0_9:.2f}ms, guaranteeing 0% SLA violation for this negotiation and below 1% overall). "
            "**To guarantee high odds for passing the Digital Twin test successfully and ensuring initial SLA compliance, you should start with a higher Edge CPU frequency proposal (e.g., around {initial_edge_cpu_proposal:.1f} GHz).** You can then gradually decrease it during negotiation if latency allows. "
            "Adjust Edge CPU frequency (aim for 25-50 GHz), ensuring no CPU allocation conflicts. "
            "Once SLA is comfortably met, actively seek configurations that maximize energy efficiency. "
            "RAN bandwidth (up to {max_ran_bw:.1f} MHz) also affects latency. "
            "Spectral efficiency (6.0-8.0) impacts transmission rate; consider this. "
        ).format(SLA_LATENCY_THRESHOLD_MS, sla_latency_threshold_ms_mult_0_9=SLA_LATENCY_THRESHOLD_MS* 0.9, max_ran_bw=max_ran_bw, initial_ran_bw_proposal=30.0, initial_edge_cpu_proposal=40.0) # Used fixed values for initial proposals in prompt

        memory_guidance_prompt = ""
        if self.collective_memory_tool: # Check if memory is enabled at all
            if self.debiased_memory_prompt_enabled:
                memory_guidance_prompt = (
                    "\n\n**Using Collective Memory (Debiased Insights):** This memory is specially designed to highlight not just successes, but also crucial **past failures (SLA violations, unresolved negotiations)**. Pay *extra* attention to these,  whether they are recent or old, as they offer invaluable lessons on what to avoid. **Prioritize avoiding negotiation conflicts and latency SLA violations.** Your debiased memory provides you the confidence to push for better energy efficiency because you understand the true risks of past failures. Do not simply gravitate towards the most frequent or seemingly 'best' past outcomes. Instead, actively explore the different trade-offs and contexts presented by both successful and failed examples. Be mindful of cognitive biases like **confirmation bias** (only seeking information that confirms your initial ideas) and the **availability heuristic** (over-relying on easily recalled successes). Your memory has been debiased to help you see a more complete picture. Focus on **why** a strategy succeeded or failed in its specific context, and **how** that learning applies to the *current* network conditions and your negotiation objectives. Adapt, don not just replicate. "
                )
            else: # Memory is enabled but not debiased
                memory_guidance_prompt = (
                    "\n\n**Using Collective Memory:** Reason over relevant memories to guide your current proposals. Adapt your proposals based on the current network conditions. **Prioritize avoiding negotiation conflicts and ensuring low latency violation.**"
                )

        tool_usage_prompt = (
            "\n\n**Tool Usage:** Before finalizing a PROPOSE_ACTION or ACCEPT_AGREEMENT that involves a new configuration, you MUST internally test your proposed RAN bandwidth and Edge CPU frequency on your Digital Twin to ensure it meets the SLA (latency < {0}ms). If your internal test reveals an SLA violation, you MUST adjust your proposal and re-test until it passes or you determine no viable solution exists. Once a proposal passes the Digital Twin test, provide your reasoning for the proposal. The system will then append the predicted latency and energy results from your Digital Twin test to your reason. "
            "**IMPORTANT: The 'reason' field MUST be a plain text string, NOT a nested JSON object, dictionary, or any other structured format. Example: 'This proposal balances energy and latency.'**"
            "**Ensure any double quotes within this plain text reason string are STRICTLY ESCAPED using a backslash (e.g., `\\\"This is an \\\\\"example\\\\\" text with escaped quotes.\\\"` or `\\\"Configuration: \\\\\"RAN BW 39.2 MHz, Edge CPU 50.0 GHz.\\\\\"\"`). Do NOT attempt to embed JSON or dictionary-like structures within the 'reason' string.**"
            "\n\n**Response Format (Strict):**"
            "\n`PROPOSE_ACTION: {{\"ran_bandwidth_mhz\": X.X, \"edge_cpu_frequency_ghz\": Y.Y, \"reason\": \"Your detailed reasoning here. This must be a plain text string, NOT a nested JSON object or any other structured format. Example: 'This proposal balances energy and latency.'\"}}`"
            "\n`ACCEPT_AGREEMENT: {{\"ran_bandwidth_mhz\": X.X, \"edge_cpu_frequency_ghz\": Y.Y, \"reason\": \"Your detailed reasoning for acceptance. This must be a plain text string, NOT a nested JSON object or any other structured format. Example: 'This proposal balances energy and latency.'\"}}`"
            "\n`NO_AGREEMENT_POSSIBLE`"
            "\n**Ensure numerical values are correct (e.g., 25.0, 35.5) and JSON is valid.** If the other agent hasn't proposed, your first move should be PROPOSE_ACTION. Do not call `enforce_actions` directly."
        ).format(SLA_LATENCY_THRESHOLD_MS)

        self.chat_session.history.append({
            "role": "user",
            "parts": [
                {"text": initial_prompt_text_base + tool_usage_prompt}
            ]
        })
