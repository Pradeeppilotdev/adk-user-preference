# 🚀 Semantic Tag Extraction & Learning System
A modern, context-aware platform for extracting, matching, and learning semantic tags from natural language. Designed for smart matchmaking, user profiling, and intelligent data enrichment, this system leverages advanced NLP and embeddings to turn free-form text into structured, actionable data.
---
> **🎯 Mission:**
> Transform unstructured language into meaningful, structured tags—automatically, accurately, and with the ability to learn and adapt.
---
## ✨ Features
- 🧠 **Semantic Tag Extraction:** Understands and extracts relevant tags from user descriptions using NLP, regex, and embeddings.
- 🤖 **Context-Aware Learning:** Learns new, meaningful phrases and expands tag lists in real time.
- 🏷️ **Category Validation:** Ensures tags are assigned to the correct category (e.g., only valid languages in the language tag).
- 🔍 **Flexible Matching:** Supports both exact and fuzzy matching for nuanced values like age and height.
- 🧹 **Deduplication & Cleaning:** Includes tools to keep tag CSVs clean and free of duplicates.
- 🌍 **Entity Recognition:** Detects locations, organizations, and more using spaCy.
- 📜 **Comprehensive Logging:** Tracks all learned phrases and semantic matches for transparency and review.
---
## 🗂️ Directory Structure
```
.
├── tag.py                  # Main script: extraction, matching, learning
├── clean.py                # Cleans and deduplicates tag CSVs
├── tags/                   # Tag CSV files (one per category)
│   ├── age.csv
│   ├── diet_lifestyle.csv
│   ├── education.csv
│   ├── gender.csv
│   ├── height.csv
│   ├── hobbies_interests.csv
│   ├── language.csv
│   ├── location.csv
│   ├── profession.csv
│   ├── relocation.csv
│   ├── religion_caste.csv
│   ├── smoking_drinking habits.csv
│   ├── spiritual_religious.csv
│   └── values_personality_traits.csv
├── tag_embeddings.pkl      # Cached tag embeddings for fast semantic search
├── tag_embeddings.hash     # Hash of tag CSVs for cache validation
├── embedding_cache.pkl     # Embedding cache for phrases
├── learned_log.txt         # Log of learned phrases
├── semantic_log.txt        # Log of semantic matches
```
---
## ⚙️ Requirements
- Python 3.8+
- [spaCy](https://spacy.io/) (`en_core_web_sm` model)
- [sentence-transformers](https://www.sbert.net/)
- torch
Install dependencies:
```bash
pip install spacy sentence-transformers torch
python -m spacy download en_core_web_sm
```
---
## 🔄 Workflow
1. **Load Tags:** The system loads all tag values from the `tags/` directory.
2. **User Input:** You provide a natural language description.
3. **Extraction:** The system uses NLP, regex, and entity recognition to extract candidate phrases.
4. **Matching:** Phrases are matched to tag categories using both direct string matching and semantic similarity.
5. **Validation:** Matches are filtered to avoid generic or irrelevant tags.
6. **Learning:** New, meaningful phrases are added to the appropriate CSV and logged.
7. **Output:** The system displays matched tags and any new learned phrases.
---
## 🏷️ Tag Categories
Each CSV in the `tags/` directory represents a tag category. The main categories are:
- `age.csv`
- `diet_lifestyle.csv`
- `education.csv`
- `gender.csv`
- `height.csv`
- `hobbies_interests.csv`
- `language.csv`
- `location.csv`
- `profession.csv`
- `relocation.csv`
- `religion_caste.csv`
- `smoking_drinking habits.csv`
- `spiritual_religious.csv`
- `values_personality_traits.csv`
You can expand these files with new values as the system learns.
---
## 🌱 Learning & Maintenance
- **Automatic Learning:** When a new, meaningful phrase is detected, it is added to the appropriate tag file and logged.
- **Semantic Assignment:** The system uses semantic similarity to assign new phrases to the best-matching category.
- **Review:** All learning is logged in `learned_log.txt` for transparency and manual review.
- **Cleaning:** Run `python clean.py` regularly to keep your tag lists high-quality and deduplicated.
- **Customization:** Add new categories by creating a new CSV in `tags/` and updating the code if needed.
---
## 💡 Tips for Best Results
> - Use clear, descriptive sentences in your input.
> - Keep your tag CSVs up to date and clean.
> - Review the logs to catch and correct any questionable matches.
> - Expand your tag CSVs with real-world, nuanced values for better accuracy.
---
## ❓ FAQ
**Q: Can I add my own tag categories?**
A: Yes! Just add a new CSV in the `tags/` directory and update the code if you want special handling.
**Q: How does the system learn new phrases?**
A: When it encounters a meaningful phrase not in the CSV, it uses semantic similarity to assign it to a category, adds it to the CSV, and logs the event.
**Q: What if the system makes a mistake?**
A: You can manually edit the CSVs or logs to correct or remove any incorrect matches or learned phrases.
**Q: How do I keep the system accurate?**
A: Regularly run `python clean.py` and review the logs. Expand your CSVs with new, relevant values as needed.
---
## 🛠️ Troubleshooting
- If you see generic or incomplete matches, run `python clean.py` to clean your CSVs.
- If the system misses a tag, check if the value exists in the relevant CSV or if the input phrasing is unusual.
- For best results, keep your tag CSVs up to date and clean.
---
## 📄 License
This project is for personal or research use. For commercial use, please contact the author.
---- Built with [spaCy](https://spacy.io/) and [sentence-transformers](https://www.sbert.net/).

