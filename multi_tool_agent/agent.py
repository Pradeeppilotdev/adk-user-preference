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


async def extract_with_advanced_ner(text: str, conversation_context: str = "") -> Dict[str, Any]:
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
        # Combine current text with conversation context for better extraction
        full_text = f"{conversation_context}\n{text}" if conversation_context else text
        
        # Use the advanced NER pipeline directly
        import os
        from dotenv import load_dotenv
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        extractor = MatrimonyProfileExtractor(openai_api_key=openai_api_key)
        result = await extractor.extract_profile(full_text)

        # Build normalized tags schema from advanced NER results
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
            if value is None or not value:
                return
            if isinstance(value, str):
                v = value.strip()
                if v and v.lower() not in [x.lower() for x in tags[tag_key]]:
                    tags[tag_key].append(v)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    add(tag_key, item)

        # Map advanced NER results to our schema
        user_traits = result.get("user_traits", {})
        partner_traits = result.get("partner_traits", {})
        
        # Gender mapping - check for "bride" in text if not found in NER
        sex = partner_traits.get("sex", {}).get("value", "")
        if sex:
            if sex.lower() in ["female", "woman", "girl"]:
                add("gender", "bride")
            elif sex.lower() in ["male", "man", "boy"]:
                add("gender", "groom")
        elif "bride" in full_text.lower():
            add("gender", "bride")
        elif "groom" in full_text.lower():
            add("gender", "groom")

        # Age mapping
        min_age = partner_traits.get("minAge", {}).get("value", "")
        max_age = partner_traits.get("maxAge", {}).get("value", "")
        if min_age and max_age:
            add("age", f"{min_age}-{max_age}")
        elif min_age:
            add("age", min_age)
        elif max_age:
            add("age", max_age)

        # Education mapping - improve education detection
        education = partner_traits.get("educationLevel", {}).get("value", "")
        if education and education.lower() not in ["be", "someone"]:
            add("education", education)
        elif "postgraduate" in full_text.lower() or "masters" in full_text.lower() or "mba" in full_text.lower():
            add("education", "postgraduate degree")
        elif "degree" in full_text.lower():
            add("education", "degree")

        # Profession mapping - improve profession detection
        occupation = partner_traits.get("occupation", {}).get("value", "")
        if occupation and occupation.lower() not in ["bride", "groom", "someone"]:
            add("profession", occupation)
        
        # Extract professions from text
        text_l = full_text.lower()
        if "computer science" in text_l:
            add("profession", "computer science")
        if "economics" in text_l:
            add("profession", "economics")
        if "law" in text_l:
            add("profession", "law")
        if "engineering" in text_l:
            add("profession", "engineering")
        if "medicine" in text_l or "medical" in text_l:
            add("profession", "medicine")
        if "media" in text_l:
            add("profession", "media")
        if "design" in text_l:
            add("profession", "design")
        if "tech" in text_l or "technology" in text_l:
            add("profession", "tech")

        # Location mapping - fix location extraction
        city = partner_traits.get("city", {}).get("value", "")
        state = partner_traits.get("state", {}).get("value", "")
        
        # Filter out incorrect location extractions
        if city and not any(word in city.lower() for word in ["tamil", "brahmin", "background", "someone", "liberal", "urban", "family"]):
            add("location", city)
        if state:
            add("location", state)
        
        # Extract locations from text
        if "chennai" in text_l:
            add("location", "Chennai")
        if "coimbatore" in text_l:
            add("location", "Coimbatore")
        if "south india" in text_l:
            add("location", "South India")
        if "tamil nadu" in text_l:
            add("location", "Tamil Nadu")
        if "liberal urban family" in text_l:
            add("location", "urban")

        # Language mapping
        languages = partner_traits.get("languagesKnown", {}).get("value", "")
        if languages:
            add("language", languages)
        
        # Extract languages from text
        if "english" in text_l and "fluent" in text_l:
            add("language", "English")
        if "tamil" in text_l:
            add("language", "Tamil")

        # Diet & lifestyle mapping
        diet = partner_traits.get("dietPreferences", {}).get("value", "")
        if diet:
            add("diet_lifestyle", diet)
        
        # Extract diet preferences from text with better logic
        if "non-vegetarian" in text_l or "non vegetarian" in text_l or "non-veg" in text_l or "non veg" in text_l:
            add("diet_lifestyle", "non-vegetarian")
        elif "vegetarian" in text_l and "non-vegetarian" not in text_l and "non vegetarian" not in text_l:
            add("diet_lifestyle", "vegetarian")
        
        smoking = partner_traits.get("smokingHabits", {}).get("value", "")
        if smoking and smoking.lower() in ["no", "non-smoker"]:
            add("diet_lifestyle", "non-smoker")
        elif "non-smoker" in text_l or "non smoker" in text_l:
            add("diet_lifestyle", "non-smoker")
        
        drinking = partner_traits.get("drinkingHabits", {}).get("value", "")
        if drinking and drinking.lower() in ["no", "non-drinker"]:
            add("diet_lifestyle", "non-drinker")
        elif "non-drinker" in text_l or "non drinker" in text_l:
            add("diet_lifestyle", "non-drinker")
        elif "social drinking" in text_l or "occasional drinking" in text_l or "occasional social drinking" in text_l:
            add("diet_lifestyle", "social drinking")

        # Religion & caste mapping
        religion = partner_traits.get("religion", {}).get("value", "")
        caste = partner_traits.get("caste", {}).get("value", "")
        if religion:
            add("religion_caste", religion)
        if caste:
            add("religion_caste", caste)

        # Spiritual/religious mapping
        lifestyle = partner_traits.get("lifestyle", {}).get("value", "")
        if lifestyle and "spiritual" in lifestyle.lower():
            add("spiritual_religious", lifestyle)
        elif "spiritual but not overly religious" in text_l:
            add("spiritual_religious", "spiritual but not overly religious")

        # Hobbies & interests mapping
        hobbies = user_traits.get("hobbiesAndInterests", {}).get("value", "")
        if hobbies:
            add("hobbies_interests", hobbies)
        
        travel = user_traits.get("travelPreferences", {}).get("value", "")
        if travel:
            add("hobbies_interests", travel)
        
        sports = user_traits.get("sportsAndFitness", {}).get("value", "")
        if sports:
            add("hobbies_interests", sports)
        
        # Extract hobbies from text
        if "reading" in text_l:
            add("hobbies_interests", "reading")
        if "music" in text_l:
            add("hobbies_interests", "music")
        if "travel" in text_l:
            add("hobbies_interests", "travel")
        if "yoga" in text_l:
            add("hobbies_interests", "yoga")
        if "classical dance" in text_l or "dance" in text_l:
            add("hobbies_interests", "classical dance")

        # Values & personality mapping
        personality = partner_traits.get("personalityTraits", {}).get("value", "")
        if personality:
            add("values_personality_traits", personality)
        
        family_values = partner_traits.get("familyValuesAlignment", {}).get("value", "")
        if family_values:
            add("values_personality_traits", family_values)
        
        # Extract values from text
        if "family values" in text_l:
            add("values_personality_traits", "family values")
        if "respects elders" in text_l or "respectful of elders" in text_l:
            add("values_personality_traits", "respects elders")
        if "traditions" in text_l:
            add("values_personality_traits", "respectful of traditions")

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
        
        # Build formatted tags display (one tag per line)
        formatted_tags = []
        if any(tags.values()):  # Only show if there are any tags
            formatted_tags.append("**Extracted Preferences:**")
            for tag_name, values in tags.items():
                if values:  # Only show non-empty tags
                    formatted_tags.append(f"• {tag_name.replace('_', ' ').title()}: {', '.join(values)}")
        else:
            formatted_tags.append("No specific preferences detected in this input.")

        return {
            "status": "success",
            "summary": summary,
            "tags": tags,
            "formatted_tags": "\n".join(formatted_tags),
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
        "MANDATORY WORKFLOW: After EVERY user response, you MUST call the extract_with_advanced_ner tool to analyze their input. "
        "Greet ONCE per session (e.g., 'Hi! Great to meet you.'). Do not repeat greetings in later turns. "
        "If the user mentions 'bride', 'groom', 'male', 'female', or similar terms, consider gender captured and move to the next category. "
        "If gender is not clear from their initial message, ask: 'Are you looking for a bride or a groom?' "
        "However, ALWAYS extract any preference information the user provides in any turn (even if unrelated to the current question) and mark that category as captured. "
        "Proceed through categories in this order: gender → age → profession → location → hobbies/interests → education → diet/lifestyle → language → religion/caste → spiritual/religious → values/personality traits. Ask ONE concise question for the next missing category only. "
        "RESPONSE FORMAT: After each user response: (1) ALWAYS call extract_with_advanced_ner tool first; (2) Use the tool's 'summary' field for natural language; (3) MANDATORY: Display the tool's 'formatted_tags' field exactly as returned (shows each tag on its own line with bullet points); (4) Ask the next question. "
        "Never loop on the same category once captured unless the user changes it. Keep prompts short if user sends fillers like 'hey'. "
        "When all categories are covered or the user says they're done: (1) call extract_with_advanced_ner one final time; (2) use its summary and tags for the final output. "
        "Do NOT include emojis or confidence scores. ALWAYS use the tool results, never generate tags manually. "
        "EXAMPLE: After calling the tool, your response should look like: 'I see you're looking for a bride. [Tool's formatted_tags output here] What age range are you looking for?'"
    ),
    tools=[extract_with_advanced_ner],
    output_key="response",
)

simple_tool = agent_tool.AgentTool(agent=simple_agent)


# Expose SimplePreferenceAgent directly as the root agent to avoid instruction interference
root_agent = simple_agent


