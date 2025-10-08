import re
import json
import spacy
from typing import Dict, List, Any
from dataclasses import dataclass
import asyncio
from openai import OpenAI
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class ExtractedEntity:
    """Represents an extracted entity with confidence score"""
    value: str
    confidence: float
    source: str  # 'rule_based', 'spacy', 'llm'

@dataclass
class ConfidenceFactors:
    """Holds various factors that contribute to confidence calculation"""
    source_reliability: float = 0.0
    entity_validity: float = 0.0
    contextual_clarity: float = 0.0
    semantic_alignment: float = 0.0
    extraction_completeness: float = 0.0

class ComprehensiveConfidenceScorer:
    def __init__(self):
        """Initialize the improved confidence scoring system"""
        self.source_weights = {
            "llm": 0.85,
            "rule_based": 0.90,
            "spacy": 0.75,
            "consensus": 0.95,
            "unknown": 0.60
        }
        self.category_validators = {}  # No specific validators for new keys
        self.positive_context_indicators = {
            "hobbiesAndInterests": ["hobby", "hobbies", "interest", "enjoy", "passion", "like"],
            "travelPreferences": ["travel", "vacation", "trip", "destination"],
            "musicOrMovieTastes": ["music", "song", "movie", "film", "band"],
            "sportsAndFitness": ["sports", "fitness", "gym", "exercise", "yoga", "football", "cricket"],
            "socialCausesOfInterest": ["charity", "volunteer", "ngo", "environment", "social cause"],
            "languagesKnown": ["speak", "language", "fluent", "mother tongue", "linguistic"],
            "cuisinePreferences": ["cuisine", "food", "vegetarian", "non veg", "non-veg", "spicy"],
            "habits": ["habit", "routine", "lifestyle", "diet", "smoke", "alcohol"],
            "dietPreferences": ["vegetarian", "vegan", "non veg", "non-vegetarian"],
            "lifeGoals": ["ambition", "dream", "goal", "aspiration", "family", "career"],
            "relocation": ["relocate", "move", "settle", "shift", "city", "state"],
            "minAge": ["age", "years", "old", "birth", "born"],
            "maxAge": ["age", "years", "old", "born"],
            "minHeight": ["height", "tall", "feet", "inch"],
            "maxHeight": ["height", "feet", "inch"],
            "minWeight": ["kg", "kilogram", "weight"],
            "maxWeight": ["kg", "kilogram", "weight"],
            "sex": ["male", "female", "man", "woman"],
            "educationLevel": ["graduate", "studied", "degree", "mba", "phd", "college", "university"],
            "occupation": ["engineer", "doctor", "teacher", "manager", "lawyer", "architect", "professor", "scientist", "developer", "job"],
            "state": ["state", "province", "region", "living in"],
            "city": ["city", "town", "village", "living in"],
            "religion": ["hindu", "muslim", "christian", "sikh", "jain", "buddhist"],
            "caste": ["brahmin", "kshatriya", "vaishya", "shudra", "community", "caste"],
            "lifestyle": ["traditional", "modern", "simple", "active", "reserved", "outgoing", "fit"],
            "personalityTraits": ["kind", "caring", "funny", "humble", "ambitious", "loyal", "gentle", "calm", "outgoing", "shy"],
            "lifestyleCompatibility": ["compatibility", "compatible", "similar"],
            "familyValuesAlignment": ["family oriented", "traditional", "liberal", "modern", "open-minded", "spiritual"],
            "sharedInterests": ["shared", "mutual", "together", "common"],
            "longTermGoals": ["future", "long term", "ambition", "plans", "dream"],
            "incomeRange": ["salary", "earning", "lpa", "lakh", "crore"],
            "maritalStatus": ["married", "divorced", "widow", "single", "unmarried"],
            "smokingHabits": ["smoke", "smoker", "cigarette"],
            "drinkingHabits": ["drink", "drinker", "alcohol", "beer", "wine"],
            "familyType": ["joint family", "nuclear family", "joint", "nuclear"]
        }

    def calculate_source_reliability(self, source: str, extraction_method_stats: Dict = None) -> float:
        base_score = self.source_weights.get(source, 0.60)
        if extraction_method_stats and source in extraction_method_stats:
            success_rate = extraction_method_stats[source].get("success_rate", 0.5)
            base_score = min(base_score * (0.8 + success_rate * 0.4), 0.95)
        return base_score

    def calculate_entity_validity(self, entity_value: str, tag: str) -> float:
        if not entity_value or not entity_value.strip():
            return 0.0
        score = 0.5
        word_count = len(entity_value.split())
        if 1 <= word_count <= 4:
            score += 0.1
        elif word_count > 8:
            score -= 0.1
        if not self._has_extraction_errors(entity_value):
            score += 0.2
        else:
            score -= 0.3
        if len(entity_value.strip()) >= 2 and entity_value.strip() not in [".", "?", "!", "-"]:
            score += 0.1
        return max(0.0, min(score, 1.0))

    def calculate_contextual_clarity(self, entity_value: str, tag: str, full_text: str, extraction_context: str = "") -> float:
        score = 0.5
        text_lower = full_text.lower()
        tag_base = tag
        if tag_base in self.positive_context_indicators:
            indicators = self.positive_context_indicators[tag_base]
            indicator_count = sum(1 for indicator in indicators if indicator in text_lower)
            score += min(indicator_count * 0.15, 0.5)
        if any(word in text_lower for word in ["looking for", "prefer", "want", "seeking"]):
            score += 0.1
        return max(0.0, min(score, 1.0))

    def calculate_semantic_alignment(self, entity_value: str, tag: str, full_text: str) -> float:
        # Simplified: no specific boosts for new keys
        return 0.6

    def calculate_extraction_completeness(self, entity_value: str, tag: str) -> float:
        if not entity_value or not entity_value.strip():
            return 0.0
        score = 0.7
        value = entity_value.strip()
        if not (value.endswith("...") or value.startswith("...")):
            score += 0.2
        else:
            score -= 0.3
        if self._has_proper_capitalization(value):
            score += 0.1
        words = value.lower().split()
        if len(set(words)) >= len(words) * 0.8:
            score += 0.1
        else:
            score -= 0.2
        text_tags = ["occupation", "educationLevel", "lifestyle", "personalityTraits", "sharedInterests", "longTermGoals"]
        if tag in text_tags and re.search(r'[a-zA-Z]', value):
            score += 0.1
        numeric_tags = ["minAge", "maxAge", "minHeight", "maxHeight", "minWeight", "maxWeight", "incomeRange"]
        if tag in numeric_tags and re.search(r'\d', value):
            score += 0.2
        return max(0.0, min(score, 1.0))

    def _has_extraction_errors(self, value: str) -> bool:
        if re.search(r'(.)\1{3,}', value):
            return True
        if value.count(' ') > 10:
            return True
        if re.search(r'[^\w\s\-\'".()&,]', value):
            return True
        return False

    def _has_proper_capitalization(self, value: str) -> bool:
        if not value:
            return False
        words = value.split()
        if len(words) == 1:
            return value[0].isupper() or value.islower()
        return True

    def calculate_comprehensive_confidence(self, entity_value: str, tag: str,
                                         extraction_details: Dict[str, Any],
                                         full_profile_context: Dict[str, Any]) -> float:
        if not entity_value or not entity_value.strip():
            return 0.0
        extraction_source = extraction_details.get('source', 'unknown')
        full_text = full_profile_context.get('original_text', '')
        factors = ConfidenceFactors()
        factors.source_reliability = self.calculate_source_reliability(
            extraction_source, full_profile_context.get('method_stats')
        )
        factors.entity_validity = self.calculate_entity_validity(entity_value, tag)
        factors.contextual_clarity = self.calculate_contextual_clarity(entity_value, tag, full_text, full_text)
        factors.semantic_alignment = self.calculate_semantic_alignment(entity_value, tag, full_text)
        factors.extraction_completeness = self.calculate_extraction_completeness(entity_value, tag)
        weights = {
            'source_reliability': 0.25,
            'entity_validity': 0.30,
            'contextual_clarity': 0.20,
            'semantic_alignment': 0.15,
            'extraction_completeness': 0.10
        }
        confidence = (
            factors.source_reliability * weights['source_reliability'] +
            factors.entity_validity * weights['entity_validity'] +
            factors.contextual_clarity * weights['contextual_clarity'] +
            factors.semantic_alignment * weights['semantic_alignment'] +
            factors.extraction_completeness * weights['extraction_completeness']
        )
        category_boosts = {
            "minAge": 0.1, "maxAge": 0.1,
            "minHeight": 0.1, "maxHeight": 0.1,
            "occupation": 0.1, "educationLevel": 0.05,
            "languagesKnown": 0.05, "incomeRange": 0.05
        }
        if tag in category_boosts:
            confidence += category_boosts[tag]
        if factors.source_reliability > 0.8 and factors.entity_validity > 0.7:
            confidence += 0.05
        if factors.contextual_clarity > 0.8 and factors.semantic_alignment > 0.7:
            confidence += 0.05
        confidence = max(0.0, min(confidence, 0.95))
        if all(f >= 0.5 for f in [factors.source_reliability, factors.entity_validity, factors.contextual_clarity, factors.semantic_alignment]):
            confidence = min(confidence * 1.05, 0.95)
        return round(confidence, 3)

class DeterministicExtractor:
        def __init__(self):
            """Initialize deterministic extraction patterns"""
            self.deterministic_patterns = {
                # Age patterns - highly deterministic
                'age_extraction': [
                    (r'\b(?:age[d]?\s*:?\s*)?(\d{1,2})\s*(?:years?\s*old|yrs?\s*old|years?|yrs?)\b', 'age'),
                    (r'\b(?:I\'?m|am)\s*(\d{1,2})\b', 'age'),
                    (r'\bage\s*:?\s*(\d{1,2})\b', 'age'),
                    (r'\b(\d{1,2})\s*to\s*(\d{1,2})\s*(?:years?|yrs?)\b', 'age_range'),
                    (r'\bbetween\s*(\d{1,2})\s*(?:and|-|to)\s*(\d{1,2})\b', 'age_range'),
                    (r'\blooking\s*for.*?(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s*(?:years?|yrs?)', 'partner_age_range'),
                ],
                
                # Height patterns
                'height_extraction': [
                    (r'\b(\d)\s*[\'\']\s*(\d{1,2})\s*[""]\b', 'height_ft_in'),
                    (r'\b(\d)\s*(?:feet|ft)\s*(\d{1,2})\s*(?:inches?|in)\b', 'height_ft_in'),
                    (r'\b(\d{2,3})\s*cm\b', 'height_cm'),
                    (r'\bheight\s*:?\s*(\d)\s*[\'\']\s*(\d{1,2})\b', 'height_ft_in'),
                    (r'\b(\d\'?\d+\"?)\s*(?:to|-)\s*(\d\'?\d+\"?)', 'height_range'),
                ],
                
                # Weight patterns
                'weight_extraction': [
                    (r'\b(\d{2,3})\s*kg\b', 'weight'),
                    (r'\bweight\s*:?\s*(\d{2,3})\s*kg', 'weight'),
                ],
                
                # Location patterns
                'location_extraction': [
                    (r'\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'location'),
                    (r'\bliving\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'location'),
                    (r'\bbased\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'location'),
                ],
                
                # Religion patterns
                'religion_extraction': [
                    (r'\b(Hindu|Muslim|Christian|Sikh|Jain|Buddhist|Parsi)\b', 'religion'),
                ],
                
                # Diet patterns
                'diet_extraction': [
                    (r'\b(vegetarian|veg|pure\s*veg)\b', 'diet_veg'),
                    (r'\b(non-vegetarian|non\s*veg|non-veg|meat\s*eater)\b', 'diet_nonveg'),
                    (r'\b(vegan)\b', 'diet_vegan'),
                ],
                
                # Income patterns
                'income_extraction': [
                    (r'\b(\d+)\s*(?:to|-)\s*(\d+)\s*LPA\b', 'income_range_lpa'),
                    (r'\b(\d+)\s*LPA\b', 'income_lpa'),
                    (r'\b(\d+)\s*(?:to|-)\s*(\d+)\s*(?:lakhs?|lacs?)\b', 'income_range_lakhs'),
                    (r'\b(\d+)\s*(?:lakhs?|lacs?)\b', 'income_lakhs'),
                ],
                
                # Habits patterns
                'habits_extraction': [
                    (r'\b(non-smoker|doesn\'t\s*smoke|no\s*smoking)\b', 'smoking_no'),
                    (r'\b(smoker|smokes)\b', 'smoking_yes'),
                    (r'\b(non-drinker|doesn\'t\s*drink|no\s*drinking|teetotaler)\b', 'drinking_no'),
                    (r'\b(social\s*drinker|drinks\s*socially)\b', 'drinking_social'),
                    (r'\b(drinker|drinks)\b', 'drinking_yes'),
                ],
                
                # Family type patterns
                'family_extraction': [
                    (r'\b(joint\s*family)\b', 'family_joint'),
                    (r'\b(nuclear\s*family)\b', 'family_nuclear'),
                ],
            }
            
            # Canonical value mappings for consistency
            self.value_normalizations = {
                'religion': {
                    'hindu': 'Hindu', 'muslim': 'Muslim', 'christian': 'Christian',
                    'sikh': 'Sikh', 'jain': 'Jain', 'buddhist': 'Buddhist'
                },
                'diet': {
                    'vegetarian': 'Vegetarian', 'veg': 'Vegetarian', 'pure veg': 'Vegetarian',
                    'non-vegetarian': 'Non-Vegetarian', 'non veg': 'Non-Vegetarian',
                    'vegan': 'Vegan'
                },
                'habits': {
                    'smoking_no': 'No', 'smoking_yes': 'Yes',
                    'drinking_no': 'No', 'drinking_social': 'Socially', 'drinking_yes': 'Yes'
                }
            }

        def extract_deterministic_entities(self, text: str) -> Dict[str, ExtractedEntity]:
            """Extract entities using deterministic rule-based patterns"""
            results = {}
            text_lower = text.lower()
            
            for pattern_group, patterns in self.deterministic_patterns.items():
                for pattern, extraction_type in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        if extraction_type == 'age':
                            age = int(match.group(1))
                            if 18 <= age <= 100:
                                # Check context to determine if it's user age or partner preference
                                context = text_lower[max(0, match.start()-30):match.end()+30]
                                if any(word in context for word in ['looking for', 'want', 'prefer', 'seeking']):
                                    results['minAge'] = ExtractedEntity(str(age), 0.95, 'deterministic')
                                    results['maxAge'] = ExtractedEntity(str(age), 0.95, 'deterministic')
                        
                        elif extraction_type == 'age_range' or extraction_type == 'partner_age_range':
                            age1, age2 = int(match.group(1)), int(match.group(2))
                            if 18 <= age1 <= 100 and 18 <= age2 <= 100 and age1 <= age2:
                                results['minAge'] = ExtractedEntity(str(age1), 0.95, 'deterministic')
                                results['maxAge'] = ExtractedEntity(str(age2), 0.95, 'deterministic')
                        
                        elif extraction_type == 'height_ft_in':
                            feet, inches = int(match.group(1)), int(match.group(2))
                            if 3 <= feet <= 7 and 0 <= inches < 12:
                                height_val = f"{feet}'{inches}\""
                                results['minHeight'] = ExtractedEntity(height_val, 0.95, 'deterministic')
                                results['maxHeight'] = ExtractedEntity(height_val, 0.95, 'deterministic')
                        
                        elif extraction_type == 'height_cm':
                            height_cm = int(match.group(1))
                            if 100 <= height_cm <= 250:
                                results['minHeight'] = ExtractedEntity(f"{height_cm}cm", 0.95, 'deterministic')
                                results['maxHeight'] = ExtractedEntity(f"{height_cm}cm", 0.95, 'deterministic')
                        
                        elif extraction_type == 'weight':
                            weight = int(match.group(1))
                            if 30 <= weight <= 200:
                                results['minWeight'] = ExtractedEntity(f"{weight} kg", 0.95, 'deterministic')
                                results['maxWeight'] = ExtractedEntity(f"{weight} kg", 0.95, 'deterministic')
                        
                        elif extraction_type == 'location':
                            location = match.group(1).title()
                            results['city'] = ExtractedEntity(location, 0.9, 'deterministic')
                        
                        elif extraction_type == 'religion':
                            religion = match.group(1)
                            results['religion'] = ExtractedEntity(religion, 0.95, 'deterministic')
                        
                        elif extraction_type in ['diet_veg', 'diet_nonveg', 'diet_vegan']:
                            diet_type = extraction_type.split('_')[1]
                            diet_normalized = self.value_normalizations['diet'].get(match.group(1).lower(), 
                                                                                'Vegetarian' if 'veg' in diet_type else 'Vegan')
                            results['dietPreferences'] = ExtractedEntity(diet_normalized, 0.95, 'deterministic')
                        
                        elif extraction_type in ['income_range_lpa', 'income_lpa', 'income_range_lakhs', 'income_lakhs']:
                            if 'range' in extraction_type:
                                income_val = f"{match.group(1)}-{match.group(2)} {'LPA' if 'lpa' in extraction_type else 'Lakhs'}"
                            else:
                                income_val = f"{match.group(1)} {'LPA' if 'lpa' in extraction_type else 'Lakhs'}"
                            results['incomeRange'] = ExtractedEntity(income_val, 0.9, 'deterministic')
                        
                        elif extraction_type in ['smoking_no', 'smoking_yes', 'drinking_no', 'drinking_social', 'drinking_yes']:
                            if 'smoking' in extraction_type:
                                results['smokingHabits'] = ExtractedEntity(
                                    self.value_normalizations['habits'][extraction_type], 0.95, 'deterministic')
                            else:
                                results['drinkingHabits'] = ExtractedEntity(
                                    self.value_normalizations['habits'][extraction_type], 0.95, 'deterministic')
                        
                        elif extraction_type in ['family_joint', 'family_nuclear']:
                            family_type = 'Joint Family' if 'joint' in extraction_type else 'Nuclear Family'
                            results['familyType'] = ExtractedEntity(family_type, 0.95, 'deterministic')
            
            return results

class MatrimonyProfileExtractor:
    def __init__(self, openai_api_key: str = None):
        """Initialize the extractor with all required models and patterns"""
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found.")
            self.nlp = None
        self.user_trait_keys = [
            "hobbiesAndInterests", "travelPreferences", "musicOrMovieTastes",
            "sportsAndFitness", "socialCausesOfInterest", "languagesKnown",
            "cuisinePreferences", "habits", "dietPreferences", "lifeGoals",
            "relocation"
        ]
        self.partner_trait_keys = [
            "minAge", "maxAge", "minHeight", "maxHeight", "minWeight", "maxWeight",
            "sex", "educationLevel", "occupation", "state", "city", "religion",
            "caste", "lifestyle", "personalityTraits", "lifestyleCompatibility",
            "familyValuesAlignment", "sharedInterests", "longTermGoals",
            "incomeRange", "maritalStatus", "smokingHabits", "drinkingHabits",
            "familyType"
        ]
        self.tag_categories = {}
        for key in self.user_trait_keys + self.partner_trait_keys:
            self.tag_categories[key] = {"priority": 1, "type": "user_trait" if key in self.user_trait_keys else "partner_trait"}
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize regex patterns for rule-based extraction"""
        self.age_patterns = [
            (r'\b(\d{2})\s*(?:years?\s*old|yrs?\s*old|years?|yrs?)\b', 1.0),
            (r'\bI\'?m\s*(\d{2})\b', 0.9),
            (r'\bage\s*:?\s*(\d{2})\b', 0.95),
            (r'\b(\d{2})\s*to\s*(\d{2})\s*(?:years?|yrs?)\b', 0.9),
            (r'\bbetween\s*(\d{2})\s*(?:and|-|to)\s*(\d{2})\b', 0.9)
        ]
        self.height_patterns = [
            (r'\b(\d)\s*[\'\']\s*(\d{1,2})\s*[""]\b', 0.95),
            (r'\b(\d)\s*(?:feet|ft)\s*(\d{1,2})\s*(?:inches?|in)\b', 0.9),
            (r'\b(\d{2,3})\s*cm\b', 0.95),
            (r'\bheight\s*:?\s*(\d)\s*[\'\']\s*(\d{1,2})\b', 0.9)
        ]
        self.weight_patterns = [
            (r'\b(\d{2,3})\s*kg\b', 0.9)
        ]

    def extract_rule_based(self, text: str) -> Dict[str, ExtractedEntity]:
        """Extract entities using rule-based patterns"""
        results = {}
        text_lower = text.lower()
        # Extract partner age preferences
        for pattern, confidence in self.age_patterns:
            for match in re.finditer(pattern, text_lower):
                if len(match.groups()) == 1:
                    age = int(match.group(1))
                    if 18 <= age <= 100:
                        context = text_lower[max(0, match.start()-20):match.end()+20]
                        if any(word in context for word in ['looking for', 'prefer', 'want', 'seeking']):
                            results['minAge'] = ExtractedEntity(str(age), confidence, 'rule_based')
                            results['maxAge'] = ExtractedEntity(str(age), confidence, 'rule_based')
                else:
                    age1, age2 = int(match.group(1)), int(match.group(2))
                    if 18 <= age1 <= 100 and 18 <= age2 <= 100 and age1 <= age2:
                        results['minAge'] = ExtractedEntity(str(age1), confidence, 'rule_based')
                        results['maxAge'] = ExtractedEntity(str(age2), confidence, 'rule_based')
        # Extract partner height preferences
        for pattern, confidence in self.height_patterns:
            for match in re.finditer(pattern, text_lower):
                if 'cm' in match.group(0):
                    height_cm = int(match.group(1))
                    if 100 <= height_cm <= 250:
                        val = f"{height_cm}cm"
                        results['minHeight'] = ExtractedEntity(val, confidence, 'rule_based')
                        results['maxHeight'] = ExtractedEntity(val, confidence, 'rule_based')
                else:
                    feet, inches = int(match.group(1)), int(match.group(2))
                    if 3 <= feet <= 7 and 0 <= inches < 12:
                        val = f"{feet}'{inches}\""
                        results['minHeight'] = ExtractedEntity(val, confidence, 'rule_based')
                        results['maxHeight'] = ExtractedEntity(val, confidence, 'rule_based')
        # Extract partner weight preferences
        for pattern, confidence in self.weight_patterns:
            for match in re.finditer(pattern, text_lower):
                weight = int(match.group(1))
                if 30 <= weight <= 200:
                    val = f"{weight} kg"
                    results['minWeight'] = ExtractedEntity(val, confidence, 'rule_based')
                    results['maxWeight'] = ExtractedEntity(val, confidence, 'rule_based')
        # Extract sex preference
        sex_patterns = {'Male': re.compile(r'looking for (?:a )?(male|man)\b', re.IGNORECASE),
                        'Female': re.compile(r'looking for (?:a )?(female|woman)\b', re.IGNORECASE)}
        for gender, pattern in sex_patterns.items():
            if pattern.search(text_lower):
                results['sex'] = ExtractedEntity(gender, 1.0, 'rule_based')
        # Extract education level preference
        edu_match = re.search(r'(?i)\b(MBA|B\.?E\.?|B\.?Tech|Ph\.?D|Graduate|Bachelor|Master)\b', text)
        if edu_match:
            edu = edu_match.group(0)
            results['educationLevel'] = ExtractedEntity(edu, 0.8, 'rule_based')
        # Extract occupation preference
        prof_match = re.search(r'(?i)looking for (?:a |an )?([A-Za-z]+)', text_lower)
        if prof_match:
            profession = prof_match.group(1).capitalize()
            results['occupation'] = ExtractedEntity(profession, 0.7, 'rule_based')
        # Extract city preference
        state_match = re.search(r'(?i)from\s+([A-Za-z ]+)', text_lower)
        if state_match:
            place = state_match.group(1).strip().title()
            city = place.split(',')[0]
            results['city'] = ExtractedEntity(city, 0.8, 'rule_based')
        # Extract religion preference
        rel_match = re.search(r'(?i)(Hindu|Muslim|Christian|Sikh|Jain|Buddhist)', text)
        if rel_match:
            religion = rel_match.group(1)
            results['religion'] = ExtractedEntity(religion, 0.9, 'rule_based')
        # Extract caste preference
        caste_match = re.search(r'(?i)(Brahmin|Kshatriya|Vaishya|Shudra)', text)
        if caste_match:
            caste = caste_match.group(1)
            results['caste'] = ExtractedEntity(caste, 0.9, 'rule_based')
        # Extract marital status preference
        if re.search(r'(?i)divorc(e|ed)', text):
            results['maritalStatus'] = ExtractedEntity("Divorced", 0.9, 'rule_based')
        if re.search(r'(?i)widow(ed)?', text):
            results['maritalStatus'] = ExtractedEntity("Widowed", 0.9, 'rule_based')
        # Extract family type preference
        if "joint family" in text_lower:
            results['familyType'] = ExtractedEntity("Joint Family", 0.9, 'rule_based')
        if "nuclear family" in text_lower:
            results['familyType'] = ExtractedEntity("Nuclear Family", 0.9, 'rule_based')
        # Extract smoking habits preference
        if re.search(r'(?i)non-?\s?smoker', text_lower):
            results['smokingHabits'] = ExtractedEntity("No", 0.9, 'rule_based')
        elif "smoker" in text_lower:
            results['smokingHabits'] = ExtractedEntity("Yes", 0.9, 'rule_based')
        # Extract drinking habits preference
        if re.search(r'(?i)non-?\s?drinker', text_lower):
            results['drinkingHabits'] = ExtractedEntity("No", 0.9, 'rule_based')
        elif "drinker" in text_lower:
            results['drinkingHabits'] = ExtractedEntity("Yes", 0.9, 'rule_based')
        return results

    def extract_spacy_entities(self, text: str) -> Dict[str, ExtractedEntity]:
        """Extract entities using spaCy NLP"""
        if not self.nlp:
            return {}
        results = {}
        doc = self.nlp(text)
        for ent in doc.ents:
            confidence = 0.8
            if ent.label_ == "GPE":
                city_name = ent.text.strip()
                results['city'] = ExtractedEntity(city_name, confidence, 'spacy')
            elif ent.label_ == "MONEY":
                if re.search(r'\d', ent.text):
                    results['incomeRange'] = ExtractedEntity(ent.text, confidence, 'spacy')
            elif ent.label_ == "LANGUAGE":
                results['languagesKnown'] = ExtractedEntity(ent.text, confidence, 'spacy')
        for token in doc:
            if token.pos_ == "ADJ":
                adj_text = token.text.lower()
                if adj_text in ['kind', 'caring', 'loyal', 'ambitious', 'outgoing', 'quiet', 'fun', 'romantic']:
                    if 'personalityTraits' not in results:
                        results['personalityTraits'] = ExtractedEntity(token.text, 0.6, 'spacy')
                    else:
                        results['personalityTraits'].value += f", {token.text}"
        return results
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean and validate JSON response from LLM"""
        # Remove any text before the first {
        start_idx = response_text.find('{')
        if start_idx == -1:
            return "{}"
        
        # Remove any text after the last }
        end_idx = response_text.rfind('}')
        if end_idx == -1:
            return "{}"
        
        cleaned = response_text[start_idx:end_idx+1]
        
        # Basic cleanup
        cleaned = cleaned.replace('\n', ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Fix common JSON issues
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)  # Remove trailing commas
        cleaned = cleaned.replace("'", '"')  # Replace single quotes with double quotes
        
        # Fix curly quotes and other unicode quote characters
        cleaned = cleaned.replace('"', '"').replace('"', '"')  # Replace curly quotes
        cleaned = cleaned.replace(''', "'").replace(''', "'")  # Replace curly apostrophes
        
        # Fix specific height format issues (e.g., "5"7"" -> "5'7\"")
        cleaned = re.sub(r'(\d)"(\d+)""', r"\1'\2\"", cleaned)  # Fix height format
        
        # Escape quotes within values properly
        # This regex finds quoted values and escapes internal quotes
        def escape_internal_quotes(match):
            value = match.group(1)
            # Escape any unescaped quotes within the value
            escaped_value = value.replace('"', '\\"')
            return f'"{escaped_value}"'
        
        # Apply to values only (content between quotes that are values, not keys)
        cleaned = re.sub(r':\s*"([^"]*(?:\\.[^"]*)*)"', lambda m: f': "{m.group(1)}"', cleaned)
        
        return cleaned
    
    def _aggressive_json_clean(self, response_text: str) -> str:
        """Aggressive cleaning for problematic JSON responses"""
        # Remove any text before the first {
        start_idx = response_text.find('{')
        if start_idx == -1:
            return "{}"
        
        # Remove any text after the last }
        end_idx = response_text.rfind('}')
        if end_idx == -1:
            return "{}"
        
        cleaned = response_text[start_idx:end_idx+1]
        
        # More aggressive cleaning
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Fix all quote variations
        cleaned = cleaned.replace('"', '"').replace('"', '"')  # Curly quotes
        cleaned = cleaned.replace(''', "'").replace(''', "'")  # Curly apostrophes
        cleaned = cleaned.replace('`', "'")  # Backticks
        
        # Fix height and measurement formats
        cleaned = re.sub(r'(\d+)"(\d+)""', r"\1'\2\"", cleaned)  # "5"7"" -> "5'7\""
        cleaned = re.sub(r'(\d+)\"(\d+)\"\"', r"\1'\2\"", cleaned)  # Alternative format
        
        # Remove trailing commas
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Fix common value issues
        cleaned = re.sub(r':\s*([^",}\[\]]+)([,}])', r': "\1"\2', cleaned)  # Unquoted values
        
        return cleaned

    async def extract_llm_entities(self, text: str) -> Dict[str, ExtractedEntity]:
        """Extract entities using constrained LLM approach with deterministic preprocessing"""
        if not self.openai_client:
            logger.warning("OpenAI client not initialized. Skipping LLM extraction.")
            return {}
        
        try:
            # First, extract what we can deterministically
            deterministic_extractor = DeterministicExtractor()
            deterministic_results = deterministic_extractor.extract_deterministic_entities(text)
            
            # Create a focused prompt only for non-deterministic extractions
            already_extracted = set(deterministic_results.keys())
            
            # Define what needs LLM extraction (subjective/contextual items)
            llm_extraction_targets = [
                'hobbiesAndInterests', 'travelPreferences', 'musicOrMovieTastes', 'sportsAndFitness',
                'socialCausesOfInterest', 'languagesKnown', 'cuisinePreferences', 'habits',
                'lifeGoals', 'relocation', 'lifestyle', 'personalityTraits', 'lifestyleCompatibility',
                'familyValuesAlignment', 'sharedInterests', 'longTermGoals', 'maritalStatus'
            ]
            
            # Only extract what hasn't been deterministically found
            remaining_targets = [target for target in llm_extraction_targets if target not in already_extracted]
            
            if not remaining_targets:
                return deterministic_results
            
            # Create advanced contextual prompt for remaining items
            prompt = f"""You are an expert matrimonial profile analyzer with deep understanding of Indian matrimonial contexts, cultural nuances, and implicit communication patterns. Extract comprehensive information from the user text and return it as structured JSON.

CONTEXT UNDERSTANDING:
- Users often write in casual, conversational tone with implicit meanings
- Indian matrimonial profiles contain cultural references, family dynamics, and lifestyle indicators
- Information may be scattered across sentences or implied through context
- Users may use euphemisms, indirect language, or cultural code words
- Look for both explicit statements and implicit indicators

TEXT TO ANALYZE:
"{text}"

EXTRACTION TARGETS (extract ONLY these categories if found):
{', '.join(remaining_targets)}

ADVANCED EXTRACTION GUIDELINES:

1. CONTEXTUAL INTELLIGENCE:
   - Read between the lines - extract implied information
   - Understand cultural context (joint family = traditional values, etc.)
   - Recognize euphemisms and indirect language
   - Consider the overall tone and communication style

2. CONFIDENCE SCORING (be precise):
   - 0.95: Direct, explicit statement ("I love reading books")
   - 0.90: Clear contextual indication ("Bookworm since childhood")
   - 0.85: Strong implication from context ("English literature graduate" → reading interest)
   - 0.80: Cultural/contextual clues ("Family-oriented" → family values)
   - 0.75: Moderate inference from lifestyle descriptions
   - 0.70: Weak but reasonable inference

CATEGORY-SPECIFIC EXTRACTION PATTERNS:

USER TRAITS:
- hobbiesAndInterests: Activities, sports, creative pursuits, recreational activities, passions
- travelPreferences: Travel destinations, trip types, adventure vs leisure, international vs domestic
- musicOrMovieTastes: Music genres, favorite artists, movie preferences, entertainment choices
- sportsAndFitness: Physical activities, gym habits, sports played, fitness routines, health consciousness
- socialCausesOfInterest: Volunteer work, charity involvement, social activism, environmental concerns
- languagesKnown: Known languages, mother tongue, fluency levels, multilingual abilities
- cuisinePreferences: Food preferences, cooking interests, restaurant choices, culinary exploration
- habits: Daily routines, lifestyle patterns, personal habits (positive focus)
- dietPreferences: Vegetarian/non-veg, dietary restrictions, food philosophy, health-based choices
- lifeGoals: Career ambitions, personal aspirations, life objectives, future plans
- relocation: Willingness to move, preferred cities, international openness, settlement preferences

PARTNER PREFERENCES (look for "looking for", "prefer", "want", "seeking"):
- lifestyle: Traditional vs modern outlook, activity level, social preferences
- personalityTraits: Character qualities desired (kind, caring, humorous, ambitious, etc.)
- lifestyleCompatibility: Shared values, similar interests, complementary traits
- familyValuesAlignment: Family importance, traditional vs modern family views, elder respect
- sharedInterests: Common hobbies, mutual activities, things to do together
- longTermGoals: Future planning alignment, career support, life direction compatibility
- maritalStatus: Previous marriage acceptance, divorce/widow openness, first marriage preference

ADVANCED INFERENCE PATTERNS:

LIFESTYLE INDICATORS:
- "Fitness enthusiast" → sportsAndFitness
- "Foodie" → cuisinePreferences
- "Travel lover" → travelPreferences
- "Movie buff" → musicOrMovieTastes
- "Nature lover" → hobbiesAndInterests, socialCausesOfInterest

PERSONALITY INFERENCE:
- "Family-oriented" → familyValuesAlignment
- "Independent but family-oriented" → lifestyle balance
- "Spiritual but modern" → personalityTraits + lifestyle
- "Career-focused" → lifeGoals

CULTURAL CONTEXT:
- "Traditional family" → familyValuesAlignment
- "Liberal outlook" → lifestyle preferences
- "Values-based person" → personalityTraits
- "Open to relocating" → relocation

COMPATIBILITY LANGUAGE:
- "Someone who understands" → personalityTraits
- "Shared interests" → sharedInterests
- "Similar values" → lifestyleCompatibility
- "Future together" → longTermGoals

EXAMPLES OF CONTEXTUAL EXTRACTION:

INPUT: "Love weekend getaways and trying street food. Looking for someone who enjoys simple pleasures and family time."

EXTRACTION:
{{
    "travelPreferences": {{"value": "weekend getaways", "confidence": 0.90}},
    "cuisinePreferences": {{"value": "street food, trying new foods", "confidence": 0.85}},
    "hobbiesAndInterests": {{"value": "weekend trips, food exploration", "confidence": 0.80}},
    "sharedInterests": {{"value": "simple pleasures", "confidence": 0.85}},
    "familyValuesAlignment": {{"value": "family time", "confidence": 0.85}}
}}

PATTERN RECOGNITION FOR INDIRECT INFORMATION:
- "Fitness freak" → sportsAndFitness: "fitness, gym, health-conscious lifestyle"
- "Netflix and chill person" → hobbiesAndInterests: "watching movies/series, relaxing at home"
- "Adventure seeker" → travelPreferences: "adventure travel, exploring new places"
- "Homebody" → lifestyle: "prefers staying home, indoor activities"
- "Social butterfly" → personalityTraits: "outgoing, sociable, enjoys social gatherings"

CRITICAL INSTRUCTIONS:
1. Extract ALL possible information, even if inferred from context
2. Look for patterns, implications, and cultural references
3. Consider the overall profile narrative, not just individual sentences
4. Pay attention to writing style and tone for personality insights
5. If multiple pieces of information support the same conclusion, increase confidence
6. Understand the human story behind the text

OUTPUT FORMAT - Return ONLY valid JSON:
{{
    "category_name": {{"value": "extracted_value", "confidence": 0.XX}}
}}"""

            # Use maximum determinism settings
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,        # Max determinism
                max_tokens=800,
                timeout=30
            )
            
            result_text = response.choices[0].message.content.strip()
            
            try:
                llm_data = json.loads(result_text)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed: {str(e)}. Attempting to fix...")
                
                # Try cleaning approaches
                cleaning_attempts = [
                    self._clean_json_response(result_text),
                    self._aggressive_json_clean(result_text)
                ]
                
                llm_data = None
                for cleaned_text in cleaning_attempts:
                    try:
                        llm_data = json.loads(cleaned_text)
                        break
                    except json.JSONDecodeError:
                        continue
                
                if llm_data is None:
                    logger.error("Failed to parse LLM response as JSON")
                    return deterministic_results
            
            # Process LLM results with strict validation
            llm_results = {}
            if isinstance(llm_data, dict):
                for tag, data in llm_data.items():
                    if tag in remaining_targets and isinstance(data, dict):
                        if 'value' in data and 'confidence' in data:
                            try:
                                confidence = float(data['confidence'])
                                value = str(data['value']).strip()
                                
                                # Strict validation
                                if (0.0 <= confidence <= 1.0 and 
                                    value and 
                                    value.lower() not in ['', 'null', 'none', 'n/a', 'not mentioned']):
                                    
                                    llm_results[tag] = ExtractedEntity(
                                        value=value,
                                        confidence=confidence,
                                        source='llm_constrained'
                                    )
                            except (ValueError, TypeError):
                                continue
            
            # Combine deterministic and LLM results, prioritizing deterministic
            combined_results = deterministic_results.copy()
            for tag, entity in llm_results.items():
                if tag not in combined_results:
                    combined_results[tag] = entity
            
            return combined_results
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            return deterministic_results if 'deterministic_results' in locals() else {}


    def normalize_llm_results(self, llm_results: Dict[str, ExtractedEntity]) -> Dict[str, ExtractedEntity]:
        """Normalize LLM results to match standard tag categories"""
        normalized = {}
        tag_mappings = {
            'hobbies': 'hobbiesAndInterests',
            'interests': 'hobbiesAndInterests',
            'languages': 'languagesKnown',
            'education': 'educationLevel',
            'profession': 'occupation',
            'income': 'incomeRange',
            'current_location': 'city',
            'preferred_location': 'city',
            'diet': 'dietPreferences',
            'smoking': 'smokingHabits',
            'drinking': 'drinkingHabits',
            'family_type': 'familyType',
            'marital_status': 'maritalStatus',
            'family_values': 'familyValuesAlignment',
            'personality_traits': 'personalityTraits',
            'shared_interests': 'sharedInterests',
            'long_term_goals': 'longTermGoals',
            'lifestyle': 'lifestyle',
            'other_preferences': 'personalityTraits'
        }
        for tag, entity in llm_results.items():
            if tag in self.tag_categories:
                normalized[tag] = entity
            elif tag in tag_mappings:
                mapped_tag = tag_mappings[tag]
                if mapped_tag in normalized:
                    normalized[mapped_tag].value += f", {entity.value}"
                else:
                    normalized[mapped_tag] = ExtractedEntity(entity.value, entity.confidence, entity.source)
        return normalized

    def merge_results(self, rule_results: Dict[str, ExtractedEntity], spacy_results: Dict[str, ExtractedEntity], llm_results: Dict[str, ExtractedEntity]) -> Dict[str, ExtractedEntity]:
        """Merge results from all extraction methods with improved conflict resolution"""
        merged = {}
        for tag, entity in llm_results.items():
            if tag in self.tag_categories:
                merged[tag] = entity
        for tag, entity in rule_results.items():
            if tag in self.tag_categories:
                if tag not in merged or (tag in ['minAge','maxAge','minHeight','maxHeight','minWeight','maxWeight'] and entity.confidence > 0.9):
                    merged[tag] = entity
        for tag, entity in spacy_results.items():
            if tag in self.tag_categories and tag not in merged:
                merged[tag] = entity
        return merged

    def integrate_comprehensive_scoring(self, results: Dict[str, Any], full_profile_context: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate the comprehensive confidence scoring system"""
        scorer = ComprehensiveConfidenceScorer()
        for tag, entity_data in results.items():
            if tag == "_metadata":
                continue
            value = entity_data.get("value", "")
            if value and value.strip():
                extraction_details = {
                    'context': full_profile_context.get('original_text', ''),
                    'source': entity_data.get('source', 'unknown')
                }
                new_conf = scorer.calculate_comprehensive_confidence(value, tag, extraction_details, full_profile_context)
                entity_data["confidence"] = new_conf
                if new_conf < 0.3:
                    entity_data["value"] = ""
                    entity_data["confidence"] = 0.0
        return results

    def post_process_results(self, results: Dict[str, ExtractedEntity], full_profile_context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ExtractedEntity objects to output dict and apply scoring"""
        basic_output = {}
        for tag, entity in results.items():
            if tag in self.tag_categories:
                basic_output[tag] = {"value": entity.value, "confidence": entity.confidence, "source": entity.source}
        missing_tags = set(self.tag_categories.keys()) - set(basic_output.keys())
        for missing_tag in missing_tags:
            basic_output[missing_tag] = {"value": "", "confidence": 0.0, "source": "none"}
        final_output = self.integrate_comprehensive_scoring(basic_output, full_profile_context)
        for tag, data in final_output.items():
            if tag != "_metadata" and data["confidence"] < 0.1:
                data["value"] = ""
                data["confidence"] = 0.0
        return final_output

    async def extract_profile(self, text: str) -> Dict[str, Any]:
        """Main extraction method with comprehensive confidence scoring"""
        start_time = time.time()
        try:
            rule_results = self.extract_rule_based(text)
            spacy_results = self.extract_spacy_entities(text) if self.nlp else {}
            llm_results = await self.extract_llm_entities(text)
            merged_results = self.merge_results(rule_results, spacy_results, llm_results)
            full_context = {
                'original_text': text,
                'all_extractions': {
                    'rule_based': rule_results,
                    'spacy': spacy_results,
                    'llm': llm_results
                },
                'extracted_entities': {k: v.value for k, v in merged_results.items()},
                'method_stats': {},
                'debug': False
            }
            final_output = self.post_process_results(merged_results, full_context)
            final_output["_metadata"] = {
                "processing_time_seconds": round(time.time() - start_time, 2),
                "extraction_methods_used": {
                    "rule_based": len(rule_results),
                    "spacy": len(spacy_results),
                    "llm": len(llm_results)
                },
                "total_entities_extracted": len([k for k, v in final_output.items() if k != "_metadata" and v["confidence"] > 0]),
                "confidence_scoring": "comprehensive_contextual"
            }
            user_data = {}
            partner_data = {}
            for tag, data in final_output.items():
                if tag == "_metadata":
                    continue
                if tag in self.user_trait_keys:
                    user_data[tag] = data
                elif tag in self.partner_trait_keys:
                    partner_data[tag] = data
            output = {
                "user_traits": user_data,
                "partner_traits": partner_data,
                "_metadata": final_output["_metadata"]
            }
            return output
        except Exception as e:
            logger.error(f"Profile extraction failed: {str(e)}")
            error_output = {"user_traits": {}, "partner_traits": {}, "_metadata": {}}
            for tag in self.user_trait_keys:
                error_output["user_traits"][tag] = {"value": "", "confidence": 0.0, "source": "none"}
            for tag in self.partner_trait_keys:
                error_output["partner_traits"][tag] = {"value": "", "confidence": 0.0, "source": "none"}
            error_output["_metadata"] = {
                "error": "Extraction failed",
                "message": str(e),
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
            return error_output

def create_api_endpoint():
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("API key not found. Please set OPENAI_API_KEY in your .env file.")
    
    extractor = MatrimonyProfileExtractor(openai_api_key=api_key)
    
    @app.route('/api/attribute-extractor', methods=['POST'])
    async def attr_extrctr():
        try:
            input_data = request.get_json()
            required_fields = ['user_id', 'txt']
            for field in required_fields:
                if field not in input_data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            result = await extractor.extract_profile(input_data.get("txt"))
            if result.get('error'):
                return jsonify(result), 500
            else:
                return jsonify(result), 200
        except Exception as e:
            return jsonify({'error': f'Server error: {str(e)}'}), 500
    return app

if __name__ == "__main__":
    app = create_api_endpoint()
    app.run(debug=True, host='0.0.0.0', port=8504)
