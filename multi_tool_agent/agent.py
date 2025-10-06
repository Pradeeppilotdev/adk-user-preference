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
    model="gemini-2.5-pro",
    description=(
        "Extracts user preferences into a fixed JSON schema without using local CSVs."
    ),
    instruction=(
        "You extract structured preferences from input text. Output ONLY compact JSON with keys: "
        "age, diet_lifestyle, education, hobbies_interests, language, location, profession, "
        "religion_caste, spiritual_religious, values_personality_traits. Each value is a list of strings. "
        "Use ranges for age when present (e.g., '24-28'). If unknown, output an empty list."
    ),
    output_key="llm_tags",
)

llm_extractor_tool = agent_tool.AgentTool(agent=llm_extractor)


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


root_agent = Agent(
    name="user_pref_tag_agent",
    model="gemini-2.5-pro",
    description=(
        "Agent that extracts structured user preference tags from freeform text using the local tagger."
    ),
    instruction=(
        "Prefer using the LLMTagExtractor tool to extract tags into the fixed schema. "
        "Use analyze_preferences only as a fallback/enricher. Provide concise summaries. "
        "When needed, initiate human review and apply approved edits."
    ),
    tools=[llm_extractor_tool, analyze_preferences, summarize_preferences, start_human_review, finalize_human_review],
)


