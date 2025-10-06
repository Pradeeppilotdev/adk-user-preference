import os
import sys
import json
from typing import Dict, Any, List

from google.adk.agents import Agent, LlmAgent
from google.adk.tools import agent_tool

# Ensure we can import the tag module from the existing project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TAG_DIR = os.path.join(PROJECT_ROOT, "User-Preference-Tagging", "tag")
if TAG_DIR not in sys.path:
    sys.path.insert(0, TAG_DIR)

import tag as tag_module  # type: ignore


def analyze_preferences(text: str) -> Dict[str, Any]:
    """Analyze freeform text and return matched tags and learned phrases.

    Wraps `extract_matches` and `semantic_learn` from the existing tag system.
    """
    # Load tags and embeddings once per call; for long-running servers you may cache
    tag_dict = tag_module.load_tags(tag_module.TAGS_DIR)
    all_embeddings = tag_module.load_tag_embeddings(tag_dict)

    matches, spans = tag_module.extract_matches(text, tag_dict)
    learned = tag_module.semantic_learn(text, tag_dict, spans, all_embeddings)

    # Merge learned into matches for a unified view
    for k, v in learned.items():
        matches.setdefault(k, []).extend(v)

    # Normalize lists to unique lowercase strings
    normalized: Dict[str, List[str]] = {}
    for k, vals in matches.items():
        seen = set()
        unique_vals: List[str] = []
        for v in vals:
            lv = v.strip().lower()
            if lv and lv not in seen:
                seen.add(lv)
                unique_vals.append(lv)
        normalized[k] = unique_vals

    return {
        "status": "success",
        "matches": normalized,
        "learned": learned,
    }


def summarize_preferences(text: str) -> Dict[str, Any]:
    """Return a brief natural-language summary of extracted preferences."""
    analysis = analyze_preferences(text)
    if analysis.get("status") != "success":
        return analysis
    matches: Dict[str, List[str]] = analysis.get("matches", {})
    if not matches:
        return {"status": "success", "summary": "No preferences detected."}
    parts = []
    for tag_name, values in matches.items():
        if values:
            parts.append(f"{tag_name}: {', '.join(values)}")
    return {"status": "success", "summary": "; ".join(parts)}


# LLM-only extractor agent that outputs a fixed JSON schema for tags
llm_extractor = LlmAgent(
    name="LLMTagExtractor",
    model="gemini-2.0-flash",
    description=(
        "Extracts user preferences into a fixed JSON schema without using local CSVs."
    ),
    instruction=(
        "You extract structured preferences from input text. Output ONLY compact JSON with keys: "
        "gender, age, diet_lifestyle, education, hobbies_interests, language, location, profession, "
        "religion_caste, spiritual_religious, values_personality_traits. Each value is a list of strings. "
        "Use ranges for age when present (e.g., '24-28'). If unknown, output an empty list."
    ),
    output_key="llm_tags",
)

llm_extractor_tool = agent_tool.AgentTool(agent=llm_extractor)

# Human approval agent for new keywords
human_approver = LlmAgent(
    name="HumanApprover",
    model="gemini-2.0-flash",
    description=(
        "Manages human approval workflow for new keywords and preferences."
    ),
    instruction=(
        "When new keywords are detected, present them clearly to the human for approval. "
        "Ask simple yes/no questions or for corrections. Keep it conversational and natural."
    ),
    output_key="approval_result",
)

human_approver_tool = agent_tool.AgentTool(agent=human_approver)

def start_human_review(text: str) -> Dict[str, Any]:
    """Produce a draft extraction for human-in-the-loop review.

    Returns a payload the human can edit/approve before finalizing.
    """
    draft = analyze_preferences(text)
    return {"status": "success", "draft": draft}


def finalize_human_review(approved_json: str) -> Dict[str, Any]:
    """Apply human-approved phrases to the underlying CSVs and embeddings.

    Args:
        approved_json: JSON string mapping tag -> list[str] of approved phrases.
    """
    try:
        approved: Dict[str, List[str]] = json.loads(approved_json)
    except Exception as e:
        return {"status": "error", "error_message": f"Invalid JSON: {e}"}

    tag_dict = tag_module.load_tags(tag_module.TAGS_DIR)
    all_embeddings = tag_module.load_tag_embeddings(tag_dict)

    applied: Dict[str, List[str]] = {}
    for tag_name, phrases in approved.items():
        if not isinstance(phrases, list):
            continue
        for phrase in phrases:
            p = (phrase or "").strip().lower()
            if not p:
                continue
            if p in tag_dict.get(tag_name, set()):
                # Already present
                continue
            # Validate with existing rules; allow learning where permitted
            if not tag_module.is_valid_for_tag(tag_name, p, tag_dict):
                continue
            # Persist to CSV and in-memory dict
            tag_module.save_to_csv(tag_name, p)
            tag_dict.setdefault(tag_name, set()).add(p)
            applied.setdefault(tag_name, []).append(p)
            # Update embeddings cache in-memory for running process
            emb = tag_module.get_cached_embedding(p)
            if tag_name in all_embeddings and all_embeddings[tag_name].shape[0]:
                all_embeddings[tag_name] = tag_module.torch.cat(
                    (all_embeddings[tag_name], emb.unsqueeze(0)), dim=0
                )
            else:
                all_embeddings[tag_name] = emb.unsqueeze(0)
            # Log learning
            tag_module.log_learning(p, tag_name, is_duplicate=False)

    return {"status": "success", "applied": applied}


# Smart conversation agent that automatically handles HITL
conversation_agent = LlmAgent(
    name="ConversationManager",
    model="gemini-2.0-flash",
    description=(
        "Manages natural conversations while automatically extracting and learning keywords."
    ),
    instruction=(
        "You are a friendly assistant that helps users by asking questions to understand their preferences. "
        "Your goal is to gradually learn about the user through natural conversation. "
        "Ask thoughtful questions about their preferences, lifestyle, interests, profession, location, etc. "
        "When they answer, use LLMTagExtractor to capture the information. "
        "If you detect new or unclear keywords, use HumanApprover to confirm with the user. "
        "Be conversational and curious - ask follow-up questions to learn more. "
        "Remember what they've told you and reference it in future questions. "
        "Never mention 'tools' or 'extraction' - just have natural conversations while learning."
    ),
    tools=[llm_extractor_tool, human_approver_tool, analyze_preferences, start_human_review, finalize_human_review],
    output_key="conversation_response",
)

conversation_tool = agent_tool.AgentTool(agent=conversation_agent)


# Simplified single agent approach to reduce API calls
simple_agent = LlmAgent(
    name="SimplePreferenceAgent",
    model="gemini-2.0-flash",
    description=(
        "A single agent that systematically gathers user preferences for their ideal partner/bride."
    ),
    instruction=(
        "You are a friendly assistant that helps users define their preferences for their ideal partner or bride/groom. "
        "Greet ONCE per session (e.g., 'Hi! Great to meet you.'). Do not repeat greetings in later turns. "
        "Immediately ask: 'Are you looking for a bride or a groom?' If not answered, briefly re-ask (without repeating the greeting). "
        "However, ALWAYS extract any preference information the user provides in any turn (even if unrelated to the current question) and mark that category as captured. "
        "Proceed through categories in this order: gender → age → profession → location → hobbies/interests → education → diet/lifestyle → language → religion/caste → spiritual/religious → values/personality traits. Ask ONE concise question for the next missing category only. "
        "Never loop on the same category once captured unless the user changes it. Keep prompts short if user sends fillers like 'hey'. "
        "When all categories are covered or the user says they're done: (1) provide a short, natural-language summary of the partner preferences; (2) call the LLMTagExtractor tool with a single-line summary of the captured partner preferences; (3) print the tool's returned JSON under a 'tags' section. "
        "If the tool fails, still return your summary."
    ),
    tools=[llm_extractor_tool],
    output_key="response",
)

simple_tool = agent_tool.AgentTool(agent=simple_agent)


# Expose SimplePreferenceAgent directly as the root agent to avoid instruction interference
root_agent = simple_agent


