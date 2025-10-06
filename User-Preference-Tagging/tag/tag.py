import os
import csv
import re
import pickle
import torch
import hashlib
import spacy
import string
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en.stop_words import STOP_WORDS
from difflib import SequenceMatcher

# === Constants ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TAGS_DIR = os.path.join(BASE_DIR, 'tags')
LOG_FILE = os.path.join(BASE_DIR, 'learned_log.txt')
CACHE_FILE = os.path.join(BASE_DIR, 'embedding_cache.pkl')
SEMANTIC_LOG = os.path.join(BASE_DIR, 'semantic_log.txt')
TAG_EMBEDDINGS_FILE = os.path.join(BASE_DIR, 'tag_embeddings.pkl')
TAG_EMBEDDINGS_HASH = os.path.join(BASE_DIR, 'tag_embeddings.hash')

EMBEDDING_MODEL = SentenceTransformer('intfloat/e5-base-v2')
nlp = spacy.load("en_core_web_sm")


# === Load or init embedding cache ===
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}
def get_cached_embedding(text):
    key = text.strip().lower()
    if key not in embedding_cache:
        embedding_cache[key] = EMBEDDING_MODEL.encode(key, convert_to_tensor=True)
    return embedding_cache[key]

def is_complete_noun_phrase(phrase):
    doc = nlp(phrase)
    # Accept only if the phrase is a full noun chunk and not a fragment
    if len(doc) < 2:
        return False
    for chunk in doc.noun_chunks:
        if chunk.text.strip().lower() == phrase.strip().lower():
            # Avoid fragments like 'a strong sense', prefer 'strong sense of identity'
            if chunk.root.head.pos_ in {"VERB", "ADJ", "NOUN"} or chunk.root.dep_ == "ROOT":
                return True
    return False

def is_valid_phrase(phrase, vague_terms, min_words=1):
    # Use NLP to check for completeness and context
    if not is_complete_noun_phrase(phrase):
        return False
    if phrase in vague_terms or phrase in STOP_WORDS:
        return False
    if len(phrase.split()) < min_words:
        return False
    if len(phrase) <= 2:
        return False
    return True

# List of generic single-word verbs/values to filter from hobbies_interests and profession
GENERIC_PROFESSION = {"currently", "work", "working", "teaching", "school", "dedication", "enterprising"}
GENERIC_HOBBIES = {"searching"}

def is_valid_for_tag(tag, phrase, tag_dict):
    # Only allow values present in the CSV for that tag
    # For learning, allow if it's a noun phrase and not too generic
    if tag == "language" and phrase == "modern":
        return False
    if tag == "profession" and (phrase == "working" or phrase in GENERIC_PROFESSION):
        return False
    if tag == "hobbies_interests" and phrase in GENERIC_HOBBIES:
        return False
    if phrase in tag_dict.get(tag, set()):
        return True
    # Allow learning new values for certain tags
    if tag in {"profession", "education", "location", "relocation", "spiritual_religious", "values_personality_traits", "hobbies_interests"}:
        # Accept multi-word noun phrases or contextually relevant phrases
        if len(phrase.split()) > 1:
            return True
    return False

def is_near_duplicate(phrase, existing_values):
    for val in existing_values:
        ratio = SequenceMatcher(None, phrase, val).ratio()
        if ratio > 0.85:
            return True
    return False

def compute_tag_files_hash(tags_dir):
    sha = hashlib.sha256()
    for filename in sorted(os.listdir(tags_dir)):
        if filename.endswith('.csv'):
            path = os.path.join(tags_dir, filename)
            with open(path, 'rb') as f:
                sha.update(f.read())
    return sha.hexdigest()

def load_tag_embeddings(tag_dict):
    current_hash = compute_tag_files_hash(TAGS_DIR)
    if os.path.exists(TAG_EMBEDDINGS_FILE) and os.path.exists(TAG_EMBEDDINGS_HASH):
        with open(TAG_EMBEDDINGS_HASH, 'r') as f:
            cached_hash = f.read().strip()
        if cached_hash == current_hash:
            with open(TAG_EMBEDDINGS_FILE, 'rb') as f:
                return pickle.load(f)
    print("🔁 Rebuilding tag embedding cache...")
    tag_embeddings = {
        tag: EMBEDDING_MODEL.encode(list(values), convert_to_tensor=True)
        for tag, values in tag_dict.items() if values
    }
    with open(TAG_EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(tag_embeddings, f)
    with open(TAG_EMBEDDINGS_HASH, 'w') as f:
        f.write(current_hash)
    return tag_embeddings

def load_tags(tags_dir):
    tag_dict = {}
    for filename in os.listdir(tags_dir):
        if filename.endswith('.csv'):
            tag = os.path.splitext(filename)[0]
            values = set()
            with open(os.path.join(tags_dir, filename), 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    for item in row:
                        # First add the complete item as-is
                        cleaned_item = item.strip().lower()
                        if cleaned_item and cleaned_item not in STOP_WORDS and len(cleaned_item) > 2:
                            values.add(cleaned_item)
                        
                        # Then also add individual chunks for flexibility
                        for chunk in re.split(r'[;,.\(\)\[\]]+', item):
                            cleaned = chunk.strip().lower()
                            if cleaned and cleaned not in STOP_WORDS and len(cleaned) > 2:
                                values.add(cleaned)
            tag_dict[tag] = values
    return tag_dict

def save_to_csv(tag, phrase):
    path = os.path.join(TAGS_DIR, f"{tag}.csv")
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([phrase.lower()])

def log_learning(phrase, tag, is_duplicate=False):
    if is_duplicate:
        return
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    category = tag.replace("_", " ").title()
    line = f'{timestamp} ✅ Learned: "{phrase}" → {category}'
    with open(LOG_FILE, 'a', encoding='utf-8') as log:
        log.write(line + '\n')


def log_semantic_match(phrase, tag, score):
    with open(SEMANTIC_LOG, 'a', encoding='utf-8') as log:
        log.write(f'{datetime.now()} | "{phrase}" matched "{tag}" with score {round(score, 3)}\n')

def extract_matches(user_input, tag_dict):
    matches = {}
    input_lower = user_input.lower()
    global_spans = []

    # --- Height extraction ---
    # Expanded patterns for all real-world, ambiguous, rare, informal, and range-based height expressions
    height_patterns = [
        # Standard feet & inches
        r"\b([4-7])['′]\s?(\d{1,2})[\"″]?\b",  # 5'11", 6'2"
        r"\b([4-7])\s?ft\.?\s?(\d{1,2})\s?(in|inches)?\b",  # 5 ft 10 in
        r"\b([4-7])\s?feet\s?(\d{1,2})\s?(inches|inch)?\b",  # 5 feet 8 inches
        # Inches only
        r"\b([5-8][0-9])\s?(in|inches)\b",  # 72 inches
        # Centimeters
        r"\b(1[0-9]{2}|2[0-4][0-9])\s?(cm|centimeters?)\b",  # 180 cm
        # Meters
        r"\b([1-2](?:[.,]\d{1,2}))\s?(m|meters?)\b",  # 1.75 m, 1,80 m
        # Written out: six feet two, five foot eleven
        r"\b(four|five|six|seven)\s?(feet|foot|ft)\s?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)?\b",
        # Rare/informal: five and a half feet, five eleven
        r"\b(four|five|six|seven)\s?(and a half|and half)\s?(feet|foot|ft)\b",  # five and a half feet
        r"\b(four|five|six|seven)\s?eleven\b",  # five eleven
        r"\b(\d)\s?foot[s]?\s?(\d{1,2})?\b",  # 5 foot 11, 5 foot
        # Slang: five eleven, six two
        r"\b(four|five|six|seven)\s?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",  # five eleven
        # Creative spacing/typos
        r"\b(\d)\s?['′]\s?(\d{1,2})\b",  # 5 ' 11
        # Ranges: 5'10" to 6'0", between 5'8" and 6'0"
        r"\b([4-7]['′]\d{1,2})\s?(to|\-|and|–)\s?([4-7]['′]\d{1,2})\b",  # 5'10" to 6'0"
        r"\bbetween\s([4-7]['′]\d{1,2})\s?(and|to|-)\s?([4-7]['′]\d{1,2})\b",  # between 5'8" and 6'0"
    ]
    ambiguous_height_pattern = r"\b(tall|short|average height|medium height|medium tall|medium short|about \d{1,2}|around \d{1,2}|approximately \d{1,2})\b"
    height_matches = []
    explicit_spans = []
    for pattern in height_patterns:
        for match in re.finditer(pattern, input_lower):
            span = match.span()
            if any(start < span[1] and end > span[0] for start, end in global_spans):
                continue
            matched = user_input[span[0]:span[1]]
            global_spans.append(span)
            height_matches.append(matched.strip())
            explicit_spans.append(span)
    # Only add ambiguous/approximate if no explicit match overlaps or is adjacent
    for match in re.finditer(ambiguous_height_pattern, input_lower):
        span = match.span()
        # Check for overlap or adjacency with explicit spans
        if any(abs(span[0] - e[1]) <= 1 or abs(span[1] - e[0]) <= 1 or (span[0] < e[1] and span[1] > e[0]) for e in explicit_spans):
            continue
        matched = user_input[span[0]:span[1]]
        global_spans.append(span)
        height_matches.append(matched.strip())
    if height_matches:
        matches.setdefault("height", []).extend(height_matches)

    # --- Age extraction (comprehensive like height) ---
    # Expanded patterns for all real-world, ambiguous, rare, informal, and range-based age expressions
    age_patterns = [
        # Complete phrases first (longest matches) - PRIORITY ORDER
        r"\b([1-9][0-9])-year-old\b",  # 24-year-old, 30-year-old (HIGHEST PRIORITY)
        r"\b([1-9][0-9])\s?year\s?old\b",  # 24 year old, 30 year old
        r"\b(?:somewhere\s+)?between\s([1-9][0-9])\s?(and|to)\s?([1-9][0-9])\s?(years? old|y/o|yrs? old|yo|years? of age)\b",  # between 30 and 32 years of age (with or without somewhere)
        r"\bbetween\s([1-9][0-9])\s?(and|to)\s?([1-9][0-9])\s?(years? old|y/o|yrs? old|yo|years? of age)\b",  # between 30 and 35 years of age
        r"\b([1-9][0-9])\s?(to|and)\s?([1-9][0-9])\s?(years? old|y/o|yrs? old|yo|years? of age)\b",  # 30 to 35 years old/age
        # Standard numeric age: 25 years old, 32 y/o, age 40, I'm 29
        r"\b([1-9][0-9])\s?(years? old|y/o|yrs? old|yo)\b",  # 25 years old, 32 y/o
        r"\bage\s?([1-9][0-9])\b",  # age 40
        r"\bI'?m\s?([1-9][0-9])\b",  # I'm 29
        r"\b([1-9][0-9])\s?(years? of age)\b",  # 30 years of age
        r"\b([1-9][0-9])\s?(years?)\b",  # 30 years
        r"\b([1-9][0-9])\s?(yrs?)\b",  # 30 yrs
        # Age ranges: 20-25, between 30 and 35, in my 40s
        r"\b([1-9][0-9])\s?[-–]\s?([1-9][0-9])\b",  # 20-25
        r"\bbetween\s([1-9][0-9])\s?(and|to)\s?([1-9][0-9])\b",  # between 30 and 35 (without years)
        r"\bin my (\d{2})s\b",  # in my 40s
        r"\b([1-9][0-9])s\b",  # 40s
        # Written out: twenty five years old, thirty-one years of age
        r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)([-\s]?(one|two|three|four|five|six|seven|eight|nine))?\s?(years? old|y/o|yrs? old|yo|years? of age)?\b",
        # Early/mid/late expressions: early thirties, mid 30s, late twenties
        r"\b(early|mid|middle|late)\s(twenties?|thirties?|forties?|fifties?|sixties?|seventies?|eighties?|nineties?)\b",  # early thirties
        r"\b(early|mid|middle|late)\s(\d{2})s\b",  # early 30s
        # More comprehensive age patterns
        r"\b(possibly|maybe|around|about|approximately|roughly|close to|near|in her|in his|in their)\s(early|mid|middle|late)\s(twenties?|thirties?|forties?|fifties?|sixties?|seventies?|eighties?|nineties?)\b",  # possibly in her early thirties
        r"\b(possibly|maybe|around|about|approximately|roughly|close to|near)\s(\d{2})s\b",  # possibly in her 30s
        r"\b(age|aged)\s(early|mid|middle|late)\s(twenties?|thirties?|forties?|fifties?|sixties?|seventies?|eighties?|nineties?)\b",  # age early thirties
        r"\b(age|aged)\s(\d{2})s\b",  # age 30s
        # Written ranges: between twenty and thirty, twenty to thirty-five
        r"\bbetween\s(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)([-\s]?(one|two|three|four|five|six|seven|eight|nine))?\s?(and|to)\s?(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)([-\s]?(one|two|three|four|five|six|seven|eight|nine))?\s?(years? old|y/o|yrs? old|yo|years? of age)?\b",
        r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)([-\s]?(one|two|three|four|five|six|seven|eight|nine))?\s?(to|and)\s?(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)([-\s]?(one|two|three|four|five|six|seven|eight|nine))?\s?(years? old|y/o|yrs? old|yo|years? of age)?\b",
        r"\bsomewhere between\s(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)([-\s]?(one|two|three|four|five|six|seven|eight|nine))?\s?(and|to)\s?(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)([-\s]?(one|two|three|four|five|six|seven|eight|nine))?\s?(years? old|y/o|yrs? old|yo|years? of age)?\b",
        # Approximate age: around 30, about 25, approximately 35
        r"\b(around|about|approximately|roughly|close to)\s([1-9][0-9])\b",  # around 30
        r"\b(around|about|approximately|roughly|close to)\s(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)([-\s]?(one|two|three|four|five|six|seven|eight|nine))?\b",
        # Age with context: someone in their 30s, a person aged 25
        r"\b(aged|age)\s([1-9][0-9])\b",  # aged 25
        r"\b(someone|person|girl|boy|man|woman)\s(in their|of age)\s([1-9][0-9])s?\b",  # someone in their 30s
        r"\b([1-9][0-9])\s?(year old|years old)\s?(girl|boy|man|woman|person)\b",  # 25 year old girl
        # Open/flexible age phrases (only age-specific ones)
        r"\b(whatever the age|any age|no age bar|no age limit|open age|all ages|no bar for age|no age restriction|no age preference|age doesn't matter|age is not important)\b",
        # Any age-related words that might appear
        r"\b(age|aged|aging|years?|yrs?|old|young|elderly|senior|adult|teen|teenager|adolescent|child|kid|baby|infant|toddler|preschooler|school-age|college-age|working-age|retirement-age|middle-aged|young adult|older adult|senior citizen|centenarian)\b",
        # Creative spacing/typos
        r"\b([1-9][0-9])\s?['′]\s?(years? old|y/o|yrs? old|yo)\b",  # 30 ' years old
        # Standalone numbers that could be age (with context check)
        r"\b([1-9][0-9])\b",  # 30 (will be filtered by context)
    ]
    
    age_matches = []
    explicit_spans = []
    
    # First pass: collect all explicit age matches
    for pattern in age_patterns[:-1]:  # Exclude the last pattern (standalone numbers)
        for match in re.finditer(pattern, input_lower):
            span = match.span()
            if any(start < span[1] and end > span[0] for start, end in global_spans):
                continue
            
            # Extract the complete matched phrase
            matched = user_input[span[0]:span[1]]
            
            # Handle special cases
            if matched.lower().startswith('somewhere '):
                # Find where the actual age expression starts
                age_start = matched.lower().find('between')
                if age_start != -1:
                    matched = matched[age_start:]  # Start from "between"
            
            # For X-year-old format, keep the complete phrase
            if re.match(r'\b\d{1,2}-year-old\b', matched.lower()):
                # Keep the complete "24-year-old" format
                pass
            elif re.match(r'\b\d{1,2}\s?year\s?old\b', matched.lower()):
                # Keep the complete "24 year old" format
                pass
            
            global_spans.append(span)
            age_matches.append(matched.strip())
            explicit_spans.append(span)
    
    # Second pass: check standalone numbers for age context
    for match in re.finditer(r"\b([1-9][0-9])\b", input_lower):
        span = match.span()
        if any(start < span[1] and end > span[0] for start, end in global_spans):
            continue
        
        # Check if this number has age context
        number = match.group(1)
        context_start = max(0, span[0] - 50)
        context_end = min(len(input_lower), span[1] + 50)
        context = input_lower[context_start:context_end]
        
        # Look for age-related words in context - be very comprehensive
        age_indicators = ['age', 'years', 'old', 'aged', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'twenty', 'between', 'somewhere', 'early', 'mid', 'late', 'thirties', 'twenties', 'forties', 'fifties', 'sixties', 'seventies', 'eighties', 'nineties', 'young', 'elderly', 'senior', 'adult', 'teen', 'adolescent', 'child', 'baby', 'infant', 'toddler', 'preschooler', 'school-age', 'college-age', 'working-age', 'retirement-age', 'middle-aged', 'young adult', 'older adult', 'senior citizen', 'centenarian']
        has_age_context = any(indicator in context for indicator in age_indicators)
        
        # More specific age context patterns
        specific_age_context = (
            re.search(rf"\b{number}\s?(years?|yrs?)\b", context) is not None or
            re.search(rf"\bage\s{number}\b", context) is not None or
            re.search(rf"\b{number}\s?(year old|years old)\b", context) is not None or
            re.search(rf"\b{number}\s?(to|and)\s?(\d{{2}})\b", context) is not None or
            re.search(rf"\b(\d{{2}})\s?(to|and)\s?{number}\b", context) is not None
        )
        
        # Check if it's in a range pattern
        is_in_range = (re.search(rf"\b{number}\s?(to|and|-|–)\s?(\d{{2}})\b", context) is not None or 
                      re.search(rf"\b(\d{{2}})\s?(to|and|-|–)\s?{number}\b", context) is not None)
        
        if has_age_context or is_in_range or specific_age_context:
            matched = user_input[span[0]:span[1]]
            global_spans.append(span)
            age_matches.append(matched.strip())
            explicit_spans.append(span)
    
    if age_matches:
        # Remove generic 'age' if more specific matches exist
        filtered = [m for m in age_matches if m.strip().lower() != "age"]
        if filtered:
            matches.setdefault('age', []).extend(filtered)
        else:
            matches.setdefault('age', []).extend(age_matches)

    # --- Context-aware extraction for smoking and alcohol ---
    # Improved negation/avoidance patterns to handle 'avoid alcohol and smoking', etc.
    non_smoking_patterns = [
        r"avoid[\w\s,]*smoking", r"no[\w\s,]*smoking", r"does not smoke", r"never smokes?", r"non[- ]smoking", r"doesn't smoke", r"don't smoke", r"not smoke", r"without smoking",
        r"don't[\w\s,]*smoke", r"do not[\w\s,]*smoke", r"never[\w\s,]*smoke", r"no[\w\s,]*smoking", r"non[- ]smoker"
    ]
    non_drinking_patterns = [
        r"avoid[\w\s,]*alcohol", r"no[\w\s,]*alcohol", r"does not drink", r"never drinks?", r"non[- ]drinking", r"doesn't drink", r"don't drink", r"not drink", r"without alcohol",
        r"don't[\w\s,]*drink", r"do not[\w\s,]*drink", r"never[\w\s,]*drink", r"no[\w\s,]*drinking", r"non[- ]drinker"
    ]
    input_lower = user_input.lower()
    found_non_smoking = any(re.search(p, input_lower) for p in non_smoking_patterns)
    found_non_drinking = any(re.search(p, input_lower) for p in non_drinking_patterns)

    # Remove any existing matches for 'smoking_drinking habits' before tag dict matching
    if 'smoking_drinking habits' in matches:
        del matches['smoking_drinking habits']

    # Add non-smoking/non-drinking if negation/avoidance is found
    non_habits = []
    if found_non_smoking:
        non_habits.append('don\'t smoke')
    if found_non_drinking:
        non_habits.append('don\'t drink')
    if non_habits:
        matches['smoking_drinking habits'] = non_habits

    # Only if not negated, allow normal tag dict matching for 'smoking_drinking habits'
    for tag, values in tag_dict.items():
        if tag == "height":
            continue  # Only match height via regex, not tag dict
        if tag == "age" and "age" in matches:
            continue  # Skip age tag dict matching if we already have age matches from regex
        
        # Allow semantic learning for smoking_drinking habits even if negation found
        # The semantic learning will handle context better than simple regex
        found_phrases = []
        for value in sorted(values, key=lambda x: -len(x)):
            pattern = r'(?<!\w)' + re.escape(value) + r'(?!\w)'
            for match in re.finditer(pattern, input_lower):
                start, end = match.start(), match.end()
                # For age, be more lenient with span conflicts to ensure no exceptions
                if tag == "age":
                    # Only skip if there's a complete overlap with existing matches
                    has_complete_overlap = any(start >= s and end <= e for s, e in global_spans)
                    if has_complete_overlap:
                        continue
                    # For age, also check if the matched phrase is actually age-related
                    # Skip very generic phrases that might be in age.csv but aren't really age
                    if len(value) < 4 or value in ['age', 'old', 'young', 'adult', 'child', 'teen']:
                        # Only include these if they have proper age context
                        context_start = max(0, start - 20)
                        context_end = min(len(input_lower), end + 20)
                        context = input_lower[context_start:context_end]
                        age_context_words = ['years', 'thirties', 'twenties', 'forties', 'fifties', 'sixties', 'seventies', 'eighties', 'nineties', 'aged', 'turning', 'birth', 'born']
                        if not any(word in context for word in age_context_words):
                            continue
                else:
                    # For other tags, use strict span checking
                    if any(start < e and end > s for s, e in global_spans):
                        continue
                
                matched_phrase = user_input[start:end].strip(string.punctuation + " ")
                global_spans.append((start, end))
                if is_valid_for_tag(tag, matched_phrase.lower(), tag_dict):
                    if matched_phrase not in found_phrases:
                        found_phrases.append(matched_phrase)
        if found_phrases:
            matches[tag] = found_phrases

    return matches, global_spans

def semantic_learn(user_input, tag_dict, used_spans, all_existing_embeddings):
    vague_terms = {'open', 'available', 'nice', 'good', 'well', 'fine', 'any', 'some', 'many',
                   'looking', 'decent', 'cool', 'okay', 'basic', 'standard'}
    new_phrases = []
    doc = nlp(user_input)
    # Add NER for locations and orgs
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC", "ORG"}:
            phrase = ent.text.strip().lower()
            if phrase not in STOP_WORDS and len(phrase) > 2:
                new_phrases.append((phrase, ent.start_char, ent.end_char, ent.label_))
    # Noun chunks
    for chunk in doc.noun_chunks:
        start, end = chunk.start_char, chunk.end_char
        if any(start < e and end > s for s, e in used_spans):
            continue
        phrase = chunk.text.strip(string.punctuation + " ").lower()
        if is_valid_phrase(phrase, vague_terms, min_words=2):
            new_phrases.append((phrase, start, end, None))
    if not new_phrases:
        return {}
    learned = {}
    for phrase, _, _, ent_label in new_phrases:
        phrase_emb = get_cached_embedding(phrase)
        best_tag = None
        best_score = -1
        for tag, embeddings in all_existing_embeddings.items():
            if not embeddings.shape[0]:
                continue
            sim_scores = util.cos_sim(phrase_emb, embeddings)[0]
            max_score = float(torch.max(sim_scores))
            if max_score > best_score:
                best_score = max_score
                best_tag = tag
        # If it's a location/entity, force tag
        if ent_label in {"GPE", "LOC"}:
            best_tag = "location"
        elif ent_label == "ORG":
            best_tag = "profession"# or a new tag if you want
        if best_score > 0.70 and best_tag:
            existing_values = tag_dict[best_tag]
            if not is_near_duplicate(phrase, existing_values) and is_valid_for_tag(best_tag, phrase, tag_dict):
                learned.setdefault(best_tag, []).append(phrase)
                tag_dict[best_tag].add(phrase)
                save_to_csv(best_tag, phrase)
                log_learning(phrase, best_tag, is_duplicate=False)
                log_semantic_match(phrase, best_tag, best_score)
            else:
                log_learning(phrase, best_tag, is_duplicate=True)
    return learned

def main():
    tag_dict = load_tags(TAGS_DIR)
    all_existing_embeddings = load_tag_embeddings(tag_dict)

    print("⚡ Semantic Tag Matcher & Real-Time Learner")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input('📝 Enter your preferences or description: ')
        if user_input.strip().lower() == 'exit':
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(embedding_cache, f)
            with open(TAG_EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(all_existing_embeddings, f)
            print("👋 Exiting. All learned data and cache have been saved.")
            break

        matches, spans = extract_matches(user_input, tag_dict)
        learned = semantic_learn(user_input, tag_dict, spans, all_existing_embeddings)

        if learned:
            for tag, phrases in learned.items():
                for phrase in phrases:
                    if tag in all_existing_embeddings:
                        emb = get_cached_embedding(phrase)
                        all_existing_embeddings[tag] = torch.cat(
                            (all_existing_embeddings[tag], emb.unsqueeze(0)), dim=0
                        )

        for tag, phrases in learned.items():
            matches.setdefault(tag, []).extend(phrases)

        if matches:
            print('\n✅ Matched tags:')
            for tag, values in matches.items():
                print(f'  - {tag}: {", ".join(values)}')

        if learned:
            print('\n🧠 Learned new matches:')
            for tag, phrases in learned.items():
                print(f'  - {tag}: {", ".join(phrases)}')

        if not matches and not learned:
            print('❌ No matches found.')

        print("\n────────────────────────────────────\n")

if __name__ == '__main__':
    main()
