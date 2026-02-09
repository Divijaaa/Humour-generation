# -*- coding: utf-8 -*-
# ============================================
#pre-scored heuristics
# ============================================
import pandas as pd
import re
from collections import Counter

# Load your CSV
df = pd.read_csv('jokes_qwen_all.csv')

def check_rare_words_present(joke, rare_words):
    """Check if both rare words are in the joke"""
    joke_lower = joke.lower()
    words = rare_words.lower().split(' & ')

    if len(words) == 2:
        word1, word2 = words[0].strip(), words[1].strip()
        has_word1 = word1 in joke_lower
        has_word2 = word2 in joke_lower

        if has_word1 and has_word2:
            return 1.0
        elif has_word1 or has_word2:
            return 0.5
        else:
            return 0.0
    return 0.0

def score_length(joke):
    """Score based on joke length (prefer 50-300 chars)"""
    length = len(joke)
    if 50 <= length <= 300:
        return 1.0
    elif 30 <= length < 50 or 300 < length <= 400:
        return 0.6
    elif length < 30:
        return 0.2
    else:
        return 0.3

def score_coherence(joke):
    """Check for basic sentence structure"""
    score = 0.0

    # Has punctuation
    if any(p in joke for p in '.!?。！？'):
        score += 0.4

    # Not too many newlines
    if joke.count('\n') <= 3:
        score += 0.3

    # Has capital letters (proper sentences)
    if any(c.isupper() for c in joke):
        score += 0.3

    return score

def score_repetition(joke):
    """Penalize excessive word repetition"""
    words = joke.lower().split()
    if len(words) == 0:
        return 0.0

    unique_words = len(set(words))
    total_words = len(words)

    ratio = unique_words / total_words

    if ratio > 0.7:
        return 1.0
    elif ratio > 0.5:
        return 0.6
    else:
        return 0.3

def score_question_setup(joke, language):
    """Bonus for joke structure (setup/punchline)"""
    score = 0.0

    if language == 'en':
        # Check for question words
        if any(q in joke.lower() for q in ['what', 'why', 'how', 'who', 'where', 'when']):
            score += 0.5
        # Check for answer pattern
        if '?' in joke and any(a in joke.lower() for a in ['because', 'to ', 'it ']):
            score += 0.5

    elif language == 'es':
        if any(q in joke.lower() for q in ['qué', 'por qué', 'cómo', 'quién', 'dónde', 'cuándo']):
            score += 0.5
        if '?' in joke and any(a in joke.lower() for a in ['porque', 'para', 'es ']):
            score += 0.5

    elif language == 'zh':
        if any(q in joke for q in ['什么', '为什么', '怎么', '谁', '哪里', '什么时候']):
            score += 0.5
        if '？' in joke:
            score += 0.5

    return score

# ============================================
# MAIN SCORING FUNCTION
# ============================================

def calculate_automatic_score(row):
    """
    Calculate automatic score for a joke (0-100)
    """
    joke = row['joke_text']
    rare_words = row['rare_words']
    language = row['language']

    # Component scores
    words_score = check_rare_words_present(joke, rare_words) * 30  # 30 points
    length_score = score_length(joke) * 20  # 20 points
    coherence_score = score_coherence(joke) * 20  # 20 points
    repetition_score = score_repetition(joke) * 15  # 15 points
    structure_score = score_question_setup(joke, language) * 15  # 15 points

    total_score = (
        words_score +
        length_score +
        coherence_score +
        repetition_score +
        structure_score
    )

    return {
        'auto_score_total': round(total_score, 2),
        'words_present_score': round(words_score, 2),
        'length_score': round(length_score, 2),
        'coherence_score': round(coherence_score, 2),
        'repetition_score': round(repetition_score, 2),
        'structure_score': round(structure_score, 2)
    }

# ============================================
# RUN SCORING
# ============================================

print("🔄 Scoring all jokes...")

# Apply scoring to all rows
scores = df.apply(calculate_automatic_score, axis=1)
scores_df = pd.DataFrame(scores.tolist())

# Combine with original data
df_scored = pd.concat([df, scores_df], axis=1)

# Save results
df_scored.to_csv('jokes_qwen_with_scores.csv', index=False)

print("✅ Scoring complete! Saved to 'jokes_qwen_with_scores.csv'")

print("\n" + "="*60)
print("📊 EVALUATION SUMMARY")
print("="*60)

# Overall statistics
print(f"\n📈 Overall Statistics:")
print(f"Total jokes: {len(df_scored)}")
print(f"Mean score: {df_scored['auto_score_total'].mean():.2f}/100")
print(f"Median score: {df_scored['auto_score_total'].median():.2f}/100")
print(f"Std dev: {df_scored['auto_score_total'].std():.2f}")

# By language
print(f"\n🌍 By Language:")
for lang in df_scored['language'].unique():
    lang_df = df_scored[df_scored['language'] == lang]
    print(f"\n{lang.upper()}:")
    print(f"  Count: {len(lang_df)}")
    print(f"  Mean score: {lang_df['auto_score_total'].mean():.2f}")
    print(f"  Best score: {lang_df['auto_score_total'].max():.2f}")
    print(f"  Worst score: {lang_df['auto_score_total'].min():.2f}")

# By prompt variant
print(f"\n📝 By Prompt Variant:")
for prompt in df_scored['prompt_variant'].unique():
    prompt_df = df_scored[df_scored['prompt_variant'] == prompt]
    print(f"\n{prompt}:")
    print(f"  Count: {len(prompt_df)}")
    print(f"  Mean score: {prompt_df['auto_score_total'].mean():.2f}")

# Top 5 jokes
print(f"\n🏆 TOP 5 JOKES:")
top_jokes = df_scored.nlargest(5, 'auto_score_total')
for idx, row in top_jokes.iterrows():
    print(f"\n{row['joke_id']} (Score: {row['auto_score_total']}/100)")
    print(f"  {row['joke_text'][:100]}...")

# Bottom 5 jokes
print(f"\n❌ BOTTOM 5 JOKES (Need improvement):")
bottom_jokes = df_scored.nsmallest(5, 'auto_score_total')
for idx, row in bottom_jokes.iterrows():
    print(f"\n{row['joke_id']} (Score: {row['auto_score_total']}/100)")
    print(f"  {row['joke_text'][:100]}...")

print("\n" + "="*60)

# Install required packages
#!pip install anthropic pandas matplotlib seaborn -q

import pandas as pd
import re
from collections import Counter
import anthropic
import json
import time

# Load your CSV
df = pd.read_csv('jokes_qwen_with_scores.csv')

# ============================================
# LANGUAGE-SPECIFIC EVALUATION FUNCTIONS
# ============================================

# ENGLISH EVALUATION
def evaluate_english_joke(joke_text, rare_words):
    """
    Evaluate English joke based on rubric
    Returns scores for each dimension (1-5)
    """
    scores = {}

    # 1. Grammatical Correctness (15%)
    # Simple heuristic checks
    grammar_score = 5
    if not any(c.isupper() for c in joke_text):  # No capitals
        grammar_score -= 1
    if not any(p in joke_text for p in '.!?'):  # No punctuation
        grammar_score -= 1
    if joke_text.count(' ') < 3:  # Too short
        grammar_score -= 1
    scores['grammar'] = max(1, grammar_score)

    # 2. Phonetic/Homophonic Wordplay (20%)
    # Check for puns, wordplay
    phonetic_score = 3  # Default moderate
    joke_lower = joke_text.lower()

    # Check for common pun patterns
    if '"' in joke_text:  # Quoted wordplay
        phonetic_score = 4
    if any(word in joke_lower for word in ['sound', 'hear', 'call', 'say']):
        phonetic_score += 1

    scores['phonetic'] = min(5, phonetic_score)

    # 3. Semantic Incongruity (25%)
    # Check if rare words create surprise
    words = rare_words.lower().split(' & ')
    if len(words) == 2:
        word1, word2 = words[0].strip(), words[1].strip()
        # If both words are in joke, good incongruity
        if word1 in joke_lower and word2 in joke_lower:
            scores['incongruity'] = 4
        else:
            scores['incongruity'] = 2
    else:
        scores['incongruity'] = 3

    # 4. Semantic Appropriateness (15%)
    # Check if words are used correctly
    scores['semantic_appropriateness'] = 4  # Assume good unless obvious issues

    # 5. Coherence & Narrative Arc (15%)
    coherence_score = 3
    if '?' in joke_text:  # Has question (setup)
        coherence_score += 1
    if any(word in joke_lower for word in ['because', 'to ', 'so ']):  # Has answer
        coherence_score += 1
    scores['coherence'] = min(5, coherence_score)

    # 6. Wit & Intelligence (10%)
    wit_score = 3
    if len(joke_text) > 50:  # Longer jokes may be more thoughtful
        wit_score += 1
    if '"' in joke_text or "'" in joke_text:  # Dialog suggests wit
        wit_score += 1
    scores['wit'] = min(5, wit_score)

    return scores

# SPANISH EVALUATION
def evaluate_spanish_joke(joke_text, rare_words):
    """
    Evaluate Spanish joke based on rubric
    """
    scores = {}

    # 1. Grammatical Correctness (15%)
    grammar_score = 5
    if not any(c.isupper() for c in joke_text):
        grammar_score -= 1
    if not any(p in joke_text for p in '.!?¿¡'):
        grammar_score -= 1
    scores['grammar'] = max(1, grammar_score)

    # 2. Register Clash & Tone Contrast (22%)
    # Check for formal vs casual mix
    register_score = 3
    formal_words = ['señor', 'usted', 'estimado', 'distinguido']
    casual_words = ['tío', 'chaval', 'colega', 'tú']

    joke_lower = joke_text.lower()
    has_formal = any(w in joke_lower for w in formal_words)
    has_casual = any(w in joke_lower for w in casual_words)

    if has_formal and has_casual:
        register_score = 5
    elif has_formal or has_casual:
        register_score = 4

    scores['register_clash'] = register_score

    # 3. Diminutive/Augmentative Usage (18%)
    diminutive_score = 3
    # Check for -ito, -ita, -illo, -illa, -ón, -ona
    if re.search(r'\w+(ito|ita|illo|illa|ón|ona)\b', joke_lower):
        diminutive_score = 4
    scores['diminutive'] = diminutive_score

    # 4. Cultural References (18%)
    scores['cultural'] = 3  # Default

    # 5. Semantic Appropriateness (15%)
    scores['semantic_appropriateness'] = 4

    # 6. Coherence (12%)
    coherence_score = 3
    if '?' in joke_text or '¿' in joke_text:
        coherence_score += 1
    if any(word in joke_lower for word in ['porque', 'para', 'entonces']):
        coherence_score += 1
    scores['coherence'] = min(5, coherence_score)

    return scores

# CHINESE EVALUATION
def evaluate_chinese_joke(joke_text, rare_words):
    """
    Evaluate Chinese joke based on rubric
    """
    scores = {}

    # 1. Grammatical Correctness (12%)
    grammar_score = 5
    common_particles = ['了', '吗', '呢', '吧', '啊', '的', '地', '得']
    has_particles = any(p in joke_text for p in common_particles)
    if not has_particles:
        grammar_score -= 1
    scores['grammar'] = max(1, grammar_score)

    # 2. Homophonic Wordplay (20%)
    scores['homophonic'] = 3  # Default

    # 3. Visual/Character-Based Wordplay (20%)
    scores['visual'] = 3  # Default

    # 4. Semantic Incongruity (20%)
    words = rare_words.split(' & ')
    if len(words) == 2:
        word1, word2 = words[0].strip(), words[1].strip()
        if word1 in joke_text and word2 in joke_text:
            scores['incongruity'] = 4
        else:
            scores['incongruity'] = 2
    else:
        scores['incongruity'] = 3

    # 5. Cultural/Literary References (15%)
    # Check for idioms (成语 are usually 4 characters)
    idiom_pattern = re.findall(r'[\u4e00-\u9fa5]{4}', joke_text)
    if len(idiom_pattern) > 0:
        scores['cultural'] = 4
    else:
        scores['cultural'] = 3

    # 6. Semantic Appropriateness (10%)
    scores['semantic_appropriateness'] = 4

    # 7. Coherence (3%)
    coherence_score = 3
    if '？' in joke_text or '?' in joke_text:
        coherence_score += 1
    scores['coherence'] = min(5, coherence_score)

    return scores

# ============================================
# CALCULATE WEIGHTED SCORES
# ============================================

def calculate_weighted_score(scores, language):
    """
    Calculate final weighted score based on language rubric
    """
    if language == 'en':
        final_score = (
            scores['grammar'] * 0.15 +
            scores['phonetic'] * 0.20 +
            scores['incongruity'] * 0.25 +
            scores['semantic_appropriateness'] * 0.15 +
            scores['coherence'] * 0.15 +
            scores['wit'] * 0.10
        )

    elif language == 'es':
        final_score = (
            scores['grammar'] * 0.15 +
            scores['register_clash'] * 0.22 +
            scores['diminutive'] * 0.18 +
            scores['cultural'] * 0.18 +
            scores['semantic_appropriateness'] * 0.15 +
            scores['coherence'] * 0.12
        )

    elif language == 'zh':
        final_score = (
            scores['grammar'] * 0.12 +
            scores['homophonic'] * 0.20 +
            scores['visual'] * 0.20 +
            scores['incongruity'] * 0.20 +
            scores['cultural'] * 0.15 +
            scores['semantic_appropriateness'] * 0.10 +
            scores['coherence'] * 0.03
        )

    # Convert to 0-100 scale
    return (final_score / 5) * 100

# ============================================
# APPLY EVALUATION TO ALL JOKES
# ============================================

print("🔄 Evaluating jokes with language-specific rubrics...")

def evaluate_joke_row(row):
    """Evaluate a single joke based on its language"""
    language = row['language']
    joke_text = row['joke_text']
    rare_words = row['rare_words']

    if language == 'en':
        scores = evaluate_english_joke(joke_text, rare_words)
    elif language == 'es':
        scores = evaluate_spanish_joke(joke_text, rare_words)
    elif language == 'zh':
        scores = evaluate_chinese_joke(joke_text, rare_words)
    else:
        return {}

    # Calculate weighted score
    weighted_score = calculate_weighted_score(scores, language)
    scores['rubric_score_total'] = round(weighted_score, 2)

    return scores

# Apply to all rows
rubric_scores = df.apply(evaluate_joke_row, axis=1)
rubric_df = pd.DataFrame(rubric_scores.tolist())

# Combine with original dataframe
df_final = pd.concat([df, rubric_df], axis=1)

# Save results
df_final.to_csv('jokes_qwen_final_evaluation.csv', index=False)

print("Evaluation complete!")

# ============================================
# ENHANCED ANALYSIS
# ============================================

print("\n" + "="*60)
print("📊 FINAL EVALUATION SUMMARY (Language-Specific Rubrics)")
print("="*60)

print(f"\n📈 Overall Statistics:")
print(f"Total jokes: {len(df_final)}")
print(f"Mean Rubric Score: {df_final['rubric_score_total'].mean():.2f}/100")
print(f"Mean Auto Score: {df_final['auto_score_total'].mean():.2f}/100")

print(f"\n🌍 By Language (Rubric Scores):")
for lang in ['en', 'es', 'zh']:
    lang_df = df_final[df_final['language'] == lang]
    print(f"\n{lang.upper()}:")
    print(f"  Count: {len(lang_df)}")
    print(f"  Mean: {lang_df['rubric_score_total'].mean():.2f}")
    print(f"  Median: {lang_df['rubric_score_total'].median():.2f}")
    print(f"  Best: {lang_df['rubric_score_total'].max():.2f}")
    print(f"  Worst: {lang_df['rubric_score_total'].min():.2f}")

print(f"\n📝 By Prompt (Rubric Scores):")
for prompt in df_final['prompt_variant'].unique():
    prompt_df = df_final[df_final['prompt_variant'] == prompt]
    print(f"\n{prompt}:")
    print(f"  Mean: {prompt_df['rubric_score_total'].mean():.2f}")

print(f"\n🏆 TOP 10 JOKES BY RUBRIC SCORE:")
top_jokes = df_final.nlargest(10, 'rubric_score_total')
for idx, (_, row) in enumerate(top_jokes.iterrows(), 1):
    print(f"\n{idx}. {row['joke_id']} ({row['language'].upper()}) - Score: {row['rubric_score_total']}/100")
    print(f"   {row['joke_text'][:80]}...")

print("\n" + "="*60)

import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Score distribution by language
for idx, lang in enumerate(['en', 'es', 'zh']):
    lang_df = df_final[df_final['language'] == lang]
    axes[0, 0].hist(lang_df['rubric_score_total'], bins=20, alpha=0.6, label=lang.upper(), edgecolor='black')

axes[0, 0].set_title('Score Distribution by Language', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Rubric Score', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].legend()

# 2. Mean scores by language
lang_means = df_final.groupby('language')['rubric_score_total'].mean()
axes[0, 1].bar(lang_means.index, lang_means.values, color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')
axes[0, 1].set_title('Mean Score by Language', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Mean Rubric Score', fontsize=12)
axes[0, 1].set_ylim([0, 100])

# Add value labels on bars
for i, v in enumerate(lang_means.values):
    axes[0, 1].text(i, v + 2, f'{v:.1f}', ha='center', fontweight='bold')

# 3. Prompt comparison
prompt_data = df_final.groupby(['language', 'prompt_variant'])['rubric_score_total'].mean().unstack()
prompt_data.plot(kind='bar', ax=axes[1, 0], color=['#9b59b6', '#f39c12'], edgecolor='black')
axes[1, 0].set_title('Prompt Variant Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Mean Rubric Score', fontsize=12)
axes[1, 0].set_xlabel('Language', fontsize=12)
axes[1, 0].legend(title='Prompt', fontsize=10)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)

# 4. Box plot comparison
df_final.boxplot(column='rubric_score_total', by='language', ax=axes[1, 1])
axes[1, 1].set_title('Score Distribution Comparison', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Rubric Score', fontsize=12)
axes[1, 1].set_xlabel('Language', fontsize=12)
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'evaluation_results.png'")

# ============================================
#llm-based scoring
# ============================================
#!pip install google-generativeai
import pandas as pd
import time
import re
import os
import google.generativeai as genai
genai.configure(api_key="")
model = genai.GenerativeModel("models/gemini-2.5-pro")
PROMPT = """
You are analyzing a joke for academic research.

You will be given:
- a joke
- a word pair

A word is considered PRESENT if any inflected, conjugated, or grammatical form
of the word appears in the joke (same lexical root / lemma).
This applies especially to verbs and non-English languages (e.g., Spanish).

If BOTH words in the word_pair do NOT appear in the joke according to this rule,
then set Overall humor to 1 regardless of other qualities.

Score each dimension from 1 (very poor) to 5 (excellent).

Respond ONLY in plain text using EXACTLY this format:
Incongruity: <number>
Wit: <number>
Cultural fit: <number>
Overall humor: <number>

Word pair:
{word_pair}

Joke:
"{joke}"
"""

def safe_get_text(response):
    try:
        if not response.candidates:
            return None
        content = response.candidates[0].content
        if not content or not content.parts:
            return None
        texts = [p.text for p in content.parts if hasattr(p, "text")]
        return " ".join(texts).strip() if texts else None
    except Exception:
        return None
def parse_scores(text):
    scores = {
        "incongruity": None,
        "wit": None,
        "cultural_fit": None,
        "overall_humor": None
    }

    if not text:
        return scores

    patterns = {
        "incongruity": r"Incongruity:\s*(\d)",
        "wit": r"Wit:\s*(\d)",
        "cultural_fit": r"Cultural fit:\s*(\d)",
        "overall_humor": r"Overall humor:\s*(\d)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            scores[key] = int(match.group(1))

    return scores

def score_joke_with_retry(joke, word_pair, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                PROMPT.format(joke=joke, word_pair=word_pair),
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 200
                }
            )

            text = safe_get_text(response)
            scores = parse_scores(text)

            if all(v is not None for v in scores.values()):
                return scores

            print("Partial/empty response — retrying")

        except Exception as e:
            print(f"Gemini error: {e}")

        time.sleep(5 * (attempt + 1))

    return {
        "incongruity": None,
        "wit": None,
        "cultural_fit": None,
        "overall_humor": None
    }

INPUT_FILE = "jokes_qwen_all.csv"
OUTPUT_FILE = "jokes_qwen_all_gemini_scored.csv"

df = pd.read_csv(INPUT_FILE)

# If output already exists → resume
if os.path.exists(OUTPUT_FILE):
    df_out = pd.read_csv(OUTPUT_FILE)
    print(f"Resuming from existing file ({len(df_out)} rows)")
else:
    df_out = df.copy()
    for col in ["incongruity", "wit", "cultural_fit", "overall_humor"]:
        df_out[col] = None

SAVE_EVERY = 20   # save progress every 20 jokes
SLEEP_SECONDS = 3  # rate-limit protection

total = len(df_out)
processed = 0

print(f"\nStarting Gemini scoring for {total} jokes\n")

for idx, row in df_out.iterrows():

    # Skip already scored rows
    if pd.notna(row["overall_humor"]):
        continue

    print(f"Scoring joke {idx + 1}/{total}")

    scores = score_joke_with_retry(joke=row["joke"],word_pair=row["word_pair"])

    for k, v in scores.items():
        df_out.at[idx, k] = v

    processed += 1
    time.sleep(SLEEP_SECONDS)

    # Save checkpoint
    if processed % SAVE_EVERY == 0:
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved checkpoint at row {idx + 1}")

# ============================================
#Hybrid Scoring combining heuristics and LLM
# ============================================
import pandas as pd
import numpy as np

auto_file = "jokes_qwen_final_evaluation.csv"
ai_file = "jokes_qwen_all_gemini_scored.csv"

df_auto = pd.read_csv(auto_file)
df_ai = pd.read_csv(ai_file)
df_auto = df_auto.rename(columns={
    "incongruity": "incongruity_auto",
    "wit": "wit_auto"
})

df_ai = df_ai.rename(columns={
    "incongruity": "incongruity_ai",
    "wit": "wit_ai"
})
df = df_auto.merge(
    df_ai[
        ["id", "incongruity_ai", "wit_ai", "cultural_fit", "overall_humor"]
    ],
    on="id",
    how="left"
)
ai_cols = ["incongruity_ai", "wit_ai", "cultural_fit", "overall_humor"]
df[ai_cols] = df[ai_cols].fillna(3.0)
def compute_final_score(row):
    lang = row["lang"]

    if lang == "en":
        final = (
            row["grammar"] * 0.15 +
            row["coherence"] * 0.15 +
            row["semantic_appropriateness"] * 0.20 +
            row["incongruity_ai"] * 0.20 +
            row["wit_ai"] * 0.15 +
            row["overall_humor"] * 0.15
        )

    elif lang == "es":
        final = (
            row["grammar"] * 0.15 +
            row["register_clash"] * 0.20 +
            row["semantic_appropriateness"] * 0.15 +
            row["incongruity_ai"] * 0.20 +
            row["cultural_fit"] * 0.15 +
            row["overall_humor"] * 0.15
        )

    elif lang == "zh":
        final = (
            row["grammar"] * 0.12 +
            row["homophonic"] * 0.20 +
            row["semantic_appropriateness"] * 0.15 +
            row["incongruity_ai"] * 0.20 +
            row["cultural_fit"] * 0.18 +
            row["overall_humor"] * 0.15
        )

    else:
        return np.nan

    return final
df["final_hybrid_score"] = df.apply(compute_final_score, axis=1)
df["final_hybrid_score_100"] = df["final_hybrid_score"] * 20
df[["rubric_score_total", "overall_humor", "final_hybrid_score_100"]].corr()
df.to_csv("jokes_qwen_final_hybrid_scored.csv", index=False)
