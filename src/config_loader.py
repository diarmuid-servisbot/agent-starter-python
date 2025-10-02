import json
import logging
import os
from typing import Dict, Any

logger = logging.getLogger("agent")


def load_company_config(company_tag: str) -> Dict[str, Any]:
    """
    Load company-specific settings and prompts.

    Args:
        company_tag: The company identifier (e.g., "Escrow", "Selene")

    Returns:
        Dict containing 'instructions' and 'agent_voice'
    """
    # Load settings
    settings_path = f"cache/company_settings/{company_tag}_settings.json"
    try:
        with open(settings_path, "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        logger.error(f"Settings file not found: {settings_path}, using default")
        settings = {"settings": {"agent_voice": "aura-asteria-en"}}

    # Load prompts
    prompts_path = f"cache/prompts/{company_tag}_en_prompts.json"
    try:
        with open(prompts_path, "r") as f:
            prompts = json.load(f)
    except FileNotFoundError:
        logger.error(f"Prompts file not found: {prompts_path}, using default")
        prompts = {"Default Prompt": "You are a helpful voice AI assistant."}

    instructions = prompts.get("Default Prompt", "You are a helpful voice AI assistant.")
    agent_voice = settings["settings"].get("agent_voice", "aura-asteria-en")

    return {
        "instructions": instructions,
        "agent_voice": agent_voice,
    }


if __name__ == "__main__":
    # For testing
    import sys
    if len(sys.argv) > 1:
        tag = sys.argv[1]
    else:
        tag = os.getenv("COMPANY_TAG")
    config = load_company_config(tag)
    print(f"Company: {tag}")
    print(f"Instructions: {config['instructions'][:200]}...")
    print(f"Agent Voice: {config['agent_voice']}")
