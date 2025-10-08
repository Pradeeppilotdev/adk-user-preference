import os
import sys
import json
import asyncio
from typing import Dict, Any, List

from google.adk.agents import Agent, LlmAgent
from google.adk.tools import agent_tool

# Ensure we can import the tag module from the existing project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TAG_DIR = os.path.join(PROJECT_ROOT, "User-Preference-Tagging", "tag")
if TAG_DIR not in sys.path:
    sys.path.insert(0, TAG_DIR)

import tag as tag_module  # type: ignore

# Import the advanced NER pipeline
from .advanced_ner import MatrimonyProfileExtractor


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


async def extract_with_advanced_ner(text: str) -> Dict[str, Any]:
    """Extract preferences using the advanced NER pipeline and normalize to tags schema.

    Returns:
        {
          "status": "success",
          "summary": str,
          "tags": {<schema>},
          "metadata": {...}
        }
    """
    try:
        extractor = MatrimonyProfileExtractor()
        result = await extractor.extract_profile(text)

        # Collect raw values (partner traits take precedence over user traits)
        raw: Dict[str, Any] = {}
        for trait, data in {**result.get("user_traits", {}), **result.get("partner_traits", {})}.items():
            value = (data or {}).get("value")
            if value:
                raw[trait.lower()] = value

        # Build normalized tags schema
        tags: Dict[str, List[str]] = {
            "gender": [],
            "age": [],
            "profession": [],
            "location": [],
            "hobbies_interests": [],
            "education": [],
            "diet_lifestyle": [],
            "language": [],
            "religion_caste": [],
            "spiritual_religious": [],
            "values_personality_traits": [],
        }

        def add(tag_key: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, str):
                v = value.strip()
                if v and v.lower() not in [x.lower() for x in tags[tag_key]]:
                    tags[tag_key].append(v)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    add(tag_key, item)

        # Gender detection
        gender_candidates = [
            raw.get("partner_gender"), raw.get("gender"), raw.get("looking_for"), raw.get("occupation")
        ]
        for g in gender_candidates:
            if not g:
                continue
            gl = str(g).strip().lower()
            if gl in {"bride", "female", "woman", "girl"}:
                add("gender", "bride")
            elif gl in {"groom", "male", "man", "boy"}:
                add("gender", "groom")

        # Age range
        min_age = raw.get("minage") or raw.get("min_age") or raw.get("age_min")
        max_age = raw.get("maxage") or raw.get("max_age") or raw.get("age_max")
        if min_age and max_age:
            add("age", f"{min_age}-{max_age}")
        elif raw.get("age"):
            add("age", str(raw.get("age")))

        # Education
        for key in ["education", "educationlevel", "degree", "highest_education"]:
            if raw.get(key):
                add("education", raw.get(key))

        # Profession
        occupation_val = raw.get("occupation")
        if occupation_val and str(occupation_val).strip().lower() not in {"bride", "groom"}:
            add("profession", occupation_val)
        for key in ["profession", "job_title", "work", "industry"]:
            if raw.get(key):
                add("profession", raw.get(key))

        # Location (city/state/country)
        for key in ["city", "state", "country", "location", "hometown", "origin"]:
            if raw.get(key):
                add("location", raw.get(key))

        # Languages
        for key in ["languagesknown", "language", "languages"]:
            if raw.get(key):
                add("language", raw.get(key))

        # Diet & lifestyle
        for key in ["diet", "diet_lifestyle", "lifestyle"]:
            if raw.get(key):
                add("diet_lifestyle", raw.get(key))

        # Religion / caste
        for key in ["religion", "caste", "religion_caste", "community"]:
            if raw.get(key):
                add("religion_caste", raw.get(key))

        # Spiritual / religious
        for key in ["spiritual", "spiritual_religious", "beliefs"]:
            if raw.get(key):
                add("spiritual_religious", raw.get(key))

        # Hobbies / interests
        for key in ["hobbies", "interests", "hobbies_interests"]:
            if raw.get(key):
                add("hobbies_interests", raw.get(key))
        # Heuristics for travel & volunteering
        text_l = (text or "").lower()
        if "travel" in text_l or "travelling" in text_l or "traveling" in text_l:
            add("hobbies_interests", "traveling")
        if "volunteer" in text_l or "community work" in text_l:
            add("hobbies_interests", "volunteering")

        # Values / personality
        for key in ["values", "personality", "values_personality_traits", "traits"]:
            if raw.get(key):
                add("values_personality_traits", raw.get(key))
        # Additional heuristics from text
        for phrase in [
            ("open-minded", "open-minded"),
            ("respectful", "respectful"),
            ("traditions", "respectful of traditions"),
            ("chill", "chill")
        ]:
            if phrase[0] in text_l:
                add("values_personality_traits", phrase[1])

        # Build concise natural-language summary
        parts: List[str] = []
        if tags["gender"]:
            parts.append(f"looking for a {tags['gender'][0]}")
        if tags["age"]:
            parts.append(f"aged {tags['age'][0]}")
        if tags["location"]:
            parts.append(f"from {', '.join(tags['location'])}")
        if tags["profession"]:
            parts.append(f"profession: {', '.join(tags['profession'])}")
        if tags["education"]:
            parts.append(f"education: {', '.join(tags['education'])}")
        if tags["language"]:
            parts.append(f"speaks {', '.join(tags['language'])}")
        if tags["diet_lifestyle"]:
            parts.append(f"diet: {', '.join(tags['diet_lifestyle'])}")
        if tags["hobbies_interests"]:
            parts.append(f"interests: {', '.join(tags['hobbies_interests'])}")
        if tags["religion_caste"]:
            parts.append(f"religion/caste: {', '.join(tags['religion_caste'])}")
        if tags["spiritual_religious"]:
            parts.append(f"spiritual/religious: {', '.join(tags['spiritual_religious'])}")
        if tags["values_personality_traits"]:
            parts.append(f"values: {', '.join(tags['values_personality_traits'])}")

        summary = "; ".join(parts) if parts else "No clear partner preferences detected yet."

        return {
            "status": "success",
            "summary": summary,
            "tags": tags,
            "metadata": result.get("_metadata", {}),
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Advanced NER extraction failed: {str(e)}",
        }


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
        "A single agent that systematically gathers user preferences for their ideal partner/bride using advanced NER."
    ),
    instruction=(
        "You are a friendly assistant that helps users define their preferences for their ideal partner using advanced NER extraction. "
        "CRITICAL: After EVERY user response, you MUST call extract_with_advanced_ner to analyze their input and show a clean summary plus a JSON 'tags' object. "
        "Greet ONCE per session (e.g., 'Hi! Great to meet you.'). Do not repeat greetings in later turns. "
        "If the user mentions 'bride', 'groom', 'male', 'female', or similar terms, consider gender captured and move to the next category. "
        "If gender is not clear from their initial message, ask: 'Are you looking for a bride or a groom?' "
        "However, ALWAYS extract any preference information the user provides in any turn (even if unrelated to the current question) and mark that category as captured. "
        "Proceed through categories in this order: gender → age → profession → location → hobbies/interests → education → diet/lifestyle → language → religion/caste → spiritual/religious → values/personality traits. Ask ONE concise question for the next missing category only. "
        "RESPONSE FORMAT: After each user response: (1) Call extract_with_advanced_ner; (2) Present a short natural-language summary; (3) Print a 'tags' JSON block with the normalized schema; (4) Ask the next question. "
        "Never loop on the same category once captured unless the user changes it. Keep prompts short if user sends fillers like 'hey'. "
        "When all categories are covered or the user says they're done: (1) provide a concise final summary; (2) call extract_with_advanced_ner one final time; (3) print the final 'tags' JSON only. "
        "Do NOT include emojis or confidence scores."
    ),
    tools=[extract_with_advanced_ner],
    output_key="response",
)

simple_tool = agent_tool.AgentTool(agent=simple_agent)


# Expose SimplePreferenceAgent directly as the root agent to avoid instruction interference
root_agent = simple_agent


