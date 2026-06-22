import json
from typing import Dict, Any

def parse_agent_message(message: str) -> Dict[str, Any]:
    """
    Parses a string message from an agent into a structured dictionary.
    """
    if not isinstance(message, str):
        return {"intent": "PARSING_FAILED", "parameters": {"reason": "Message is not a string."}}

    message = message.strip()
    
    # Handle NO_AGREEMENT_POSSIBLE as a special case, as it may not have a JSON part.
    if message.startswith("NO_AGREEMENT_POSSIBLE"):
        reason_str = "No specific reason provided by the agent."
        # Check if a reason is provided after a colon
        if ':' in message:
            try:
                # Take everything after the first colon as the reason
                reason_str = message.split(':', 1)[1].strip()
            except IndexError:
                # This case is unlikely if ':' is in message, but is safe to have
                pass
        return {"intent": "NO_AGREEMENT_POSSIBLE", "parameters": {"reason": reason_str}}

    # For other intents, expect the 'INTENT: JSON' format
    if ':' not in message:
        return {"intent": "PARSING_FAILED", "parameters": {"reason": f"Invalid format. Expected 'INTENT: JSON', got '{message}'"}}
    
    intent, parameters_str = message.split(':', 1)
    intent = intent.strip()

    try:
        parameters = json.loads(parameters_str.strip())
        if not isinstance(parameters, dict):
            # The loaded JSON should be a dictionary
            raise json.JSONDecodeError("JSON content is not a dictionary/object.", parameters_str, 0)
        return {"intent": intent, "parameters": parameters}
    except json.JSONDecodeError as e:
        return {
            "intent": "PARSING_FAILED",
            "parameters": {"reason": f"JSON decoding failed: {e}. Original string: '{parameters_str.strip()}'"}
        }
    except Exception as e:
        return {
            "intent": "PARSING_FAILED",
            "parameters": {"reason": f"An unexpected error occurred during parsing: {e}"}
        }

