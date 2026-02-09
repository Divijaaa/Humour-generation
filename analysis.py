# ============================================
#Comparing Qwen and Gemma (final models)
# ============================================
import pandas as pd

qwen = pd.read_csv("jokes_qwen_final_hybrid_scored.csv")
gemma = pd.read_csv("jokes_gemma_final_hybrid_scored.csv")
qwen = qwen.rename(columns={
    "joke_id": "id",
    "language": "lang",
    "joke_text": "joke",
    "rare_words": "word_pair",
    "generated_model": "model",
    "prompt_variant": "prompt"
})
gemma = gemma.rename(columns={
    "id": "id",
    "lang": "lang",
    "joke": "joke",
    "word_pair": "word_pair",
    "model": "model",
    "prompt": "prompt"
})
qwen["source_model"] = "qwen"
gemma["source_model"] = "gemma"
analysis_cols = [
    "id", "lang", "joke", "word_pair", "model", "prompt", "source_model",
    "rubric_score_total",
    "overall_humor",
    "final_hybrid_score_100"
]

qwen_clean = qwen[analysis_cols].copy()
gemma_clean = gemma[analysis_cols].copy()
print(qwen_clean.isna().sum())
print(gemma_clean.isna().sum())
all_df = pd.concat([qwen_clean, gemma_clean], ignore_index=True)
all_df["source_model"].value_counts()
summary = (
    all_df
    .groupby("source_model")[[
        "rubric_score_total",
        "overall_humor",
        "final_hybrid_score_100"
    ]]
    .agg(["mean", "std", "median", "min", "max"])
    .round(2)
)

summary
lang_summary = (
    all_df
    .groupby(["source_model", "lang"])[
        ["rubric_score_total", "overall_humor", "final_hybrid_score_100"]
    ]
    .mean()
    .round(2)
)

lang_summary
qwen_clean["word_pair_norm"] = (
    qwen_clean["word_pair"]
    .str.lower()
    .str.replace(r"\s*&\s*", " & ", regex=True)
    .str.strip()
)
gemma_clean["word_pair_norm"] = (
    gemma_clean["word_pair"]
    .str.lower()
    .str.replace(r"\s*&\s*", " & ", regex=True)
    .str.strip()
)
for df in [qwen_clean, gemma_clean]:
    df["lang_norm"] = df["lang"].str.lower()
    df["prompt_norm"] = df["prompt"].str.lower()
def make_pair_key(row):
    return f"{row['lang_norm']}::{row['word_pair_norm']}::{row['prompt_norm']}"
qwen_clean["pair_key"] = qwen_clean.apply(make_pair_key, axis=1)
gemma_clean["pair_key"] = gemma_clean.apply(make_pair_key, axis=1)
all_df = pd.concat([qwen_clean, gemma_clean], ignore_index=True)

pivot = all_df.pivot_table(
    index="pair_key",
    columns="source_model",
    values="final_hybrid_score_100",
    aggfunc="mean"
)
pivot.head()
pivot["delta_qwen_minus_gemma"] = pivot["qwen"] - pivot["gemma"]

pivot["delta_qwen_minus_gemma"].describe().round(2)
pivot.notna().all(axis=1).mean()
pivot.sample(5)
pivot = pivot.reset_index()

pivot["lang"] = pivot["pair_key"].str.split("::").str[0]
pivot["lang"].value_counts()
lang_stats = pivot.groupby("lang")["delta_qwen_minus_gemma"].describe()
lang_stats.round(2)
win_rates = (
    pivot
    .assign(
        qwen_win = pivot["delta_qwen_minus_gemma"] > 0,
        gemma_win = pivot["delta_qwen_minus_gemma"] < 0
    )
    .groupby("language")[["qwen_win", "gemma_win"]]
    .mean()
    * 100
)

win_rates.round(1)
from scipy.stats import ttest_rel

for lang in ["en", "es", "zh"]:
    subset = pivot[pivot["language"] == lang][["qwen", "gemma"]].dropna()

    t_stat, p_val = ttest_rel(subset["qwen"], subset["gemma"])

    print(
        f"{lang.upper()}: "
        f"n={len(subset)}, "
        f"t={t_stat:.2f}, "
        f"p={p_val:.6f}"
    )
def cohens_d_paired(diff):
    return diff.mean() / diff.std(ddof=1)

for lang in ["en", "es", "zh"]:
    subset = pivot[pivot["language"] == lang]["delta_qwen_minus_gemma"].dropna()
    d = cohens_d_paired(subset)

    print(f"{lang.upper()}: Cohen's d = {d:.2f}")
# Standardize join key FIRST (important)
qwen['pair_key'] = qwen['word_pair'].str.lower().str.strip()
gemma['pair_key'] = gemma['word_pair'].str.lower().str.strip()

merged = qwen.merge(
    gemma,
    on=['pair_key', 'lang'],
    suffixes=('_qwen', '_gemma')
)
components = [
    'incongruity_ai',
    'wit_ai',
    'cultural_fit',
    'rubric_score_total',
    'final_hybrid_score_100'
]

for c in components:
    merged[f'delta_{c}'] = (
        merged[f'{c}_qwen'] - merged[f'{c}_gemma']
    )
component_summary = (
    merged
    .groupby('lang')[[f'delta_{c}' for c in components]]
    .mean()
    .round(3)
)

component_summary
for lang, df in merged.groupby('lang'):
    total = df['delta_final_hybrid_score_100'].mean()
    print(f"\n{lang.upper()} contribution %:")
    for c in components[:-1]:
        pct = 100 * df[f'delta_{c}'].mean() / total
        print(f"  {c}: {pct:.1f}%")
import pandas as pd

contrib_data = pd.DataFrame({
    "language": ["en", "es", "zh"],
    "rubric_score_total": [481.2, 46.3, 75.8],
    "incongruity_ai": [-7.7, 13.3, 3.2],
    "wit_ai": [-0.5, 16.1, 7.7],
    "cultural_fit": [93.1, 7.6, 14.1]
})

contrib_data = contrib_data.set_index("language")
contrib_data
import matplotlib.pyplot as plt

ax = contrib_data.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

ax.axhline(0, linewidth=1)  # baseline
ax.set_ylabel("Contribution to Qwen–Gemma score difference (%)")
ax.set_xlabel("Language")
ax.set_title("Contribution Breakdown of Hybrid Score Advantage")

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for nicer visuals
sns.set(style="whitegrid")

# Create boxplot
plt.figure(figsize=(8, 5))
ax = sns.boxplot(
    x='lang',
    y='delta_final_hybrid_score_100',
    data=merged,
    palette="Set2"
)

# Overlay mean and median
for i, lang in enumerate(merged['lang'].unique()):
    lang_data = merged[merged['lang'] == lang]['delta_final_hybrid_score_100']
    mean_val = lang_data.mean()
    median_val = lang_data.median()
    # Plot mean as a red dot
    ax.scatter(i, mean_val, color='red', marker='o', s=60, label='Mean' if i==0 else "")
    # Plot median as a black line on top of the box
    ax.scatter(i, median_val, color='black', marker='D', s=40, label='Median' if i==0 else "")

# Add labels and title
ax.set_title("Distribution of Qwen–Gemma Score Differences per Language")
ax.set_xlabel("Language")
ax.set_ylabel("Delta Final Hybrid Score (Qwen − Gemma)")
ax.axhline(0, color='gray', linewidth=1, linestyle='--')  # baseline

# Show legend
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

components = ['incongruity_ai', 'wit_ai', 'cultural_fit', 'rubric_score_total']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: Stacked Bar Chart ---
contrib_data[components].plot(
    kind='bar',
    stacked=True,
    ax=axes[0],
    cmap='Set2'
)
axes[0].axhline(0, linewidth=1, color='gray')
axes[0].set_ylabel("Contribution to Qwen–Gemma score difference (%)")
axes[0].set_xlabel("Language")
axes[0].set_title("Component Contribution Breakdown")
axes[0].legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left')

# --- Right: Box Plots with mean and median ---
ax = axes[1]
sns.boxplot(
    x='lang',
    y='delta_final_hybrid_score_100',
    data=merged,
    palette="Set2",
    ax=ax
)

# Overlay mean and median
for i, lang in enumerate(merged['lang'].unique()):
    lang_data = merged[merged['lang'] == lang]['delta_final_hybrid_score_100']
    mean_val = lang_data.mean()
    median_val = lang_data.median()
    ax.scatter(i, mean_val, color='red', marker='o', s=60, label='Mean' if i==0 else "")
    ax.scatter(i, median_val, color='black', marker='D', s=40, label='Median' if i==0 else "")

ax.axhline(0, color='gray', linewidth=1, linestyle='--')
ax.set_title("Distribution of Qwen–Gemma Score Differences")
ax.set_xlabel("Language")
ax.set_ylabel("Delta Final Hybrid Score")
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

for lang, df in merged.groupby('lang'):
    sns.histplot(df['delta_final_hybrid_score_100'], kde=True)
    plt.title(f"Histogram of Qwen-Gemma Differences ({lang.upper()})")
    plt.show()
from statsmodels.stats.descriptivestats import sign_test

# Loop through languages
for lang in merged['lang'].unique():
    df_lang = merged[merged['lang'] == lang]
    delta = df_lang['delta_final_hybrid_score_100']

    # Run paired sign test
    stat, p_val = sign_test(delta, mu0=0)

    # Descriptive summary
    n_pos = sum(delta > 0)
    n_neg = sum(delta < 0)
    n_zero = sum(delta == 0)

    print(f"\n=== {lang.upper()} ===")
    print(f"Positive differences (Qwen>Gemma): {n_pos}")
    print(f"Negative differences (Qwen<Gemma): {n_neg}")
    print(f"No difference: {n_zero}")
    print(f"Sign test p-value: {p_val:.6f}")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Prepare sign test effect sizes
effect_sizes = {}
for lang, df_lang in merged.groupby('lang'):
    delta = df_lang['delta_final_hybrid_score_100']
    n_pos = sum(delta > 0)
    n_neg = sum(delta < 0)
    effect_sizes[lang] = n_pos / (n_pos + n_neg)  # proportion of positive differences

# Plot
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))

ax = sns.boxplot(
    x='lang',
    y='delta_final_hybrid_score_100',
    data=merged,
    palette="Set2"
)

# Overlay mean and median
for i, lang in enumerate(merged['lang'].unique()):
    delta = merged[merged['lang'] == lang]['delta_final_hybrid_score_100']
    mean_val = delta.mean()
    median_val = delta.median()

    # Mean
    ax.scatter(i, mean_val, color='red', marker='o', s=60, label='Mean' if i==0 else "")
    # Median
    ax.scatter(i, median_val, color='black', marker='D', s=40, label='Median' if i==0 else "")

    # Annotate sign test effect size
    ax.text(
        i, delta.max() + 2,  # place slightly above max
        f"Sign effect: {effect_sizes[lang]:.2f}",
        horizontalalignment='center',
        fontsize=10,
        fontweight='bold'
    )

# Baseline at 0
ax.axhline(0, color='gray', linestyle='--', linewidth=1)

# Labels and title
ax.set_xlabel("Language")
ax.set_ylabel("Delta Final Hybrid Score (Qwen − Gemma)")
ax.set_title("Qwen vs. Gemma: Score Differences with Sign Test Effect Sizes")
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Component contributions (as percentages) for each language
contrib_data = pd.DataFrame({
    'incongruity_ai': [-7.7, 13.3, 3.2],
    'wit_ai': [-0.5, 16.1, 7.7],
    'cultural_fit': [93.1, 7.6, 14.1],
    'rubric_score_total': [481.2, 46.3, 75.8]
}, index=['EN', 'ES', 'ZH'])

# Create figure with 2 subplots: left=boxplot, right=stacked bar
fig, axes = plt.subplots(1, 2, figsize=(16,6))

# --- Left: Boxplots of delta scores ---
sns.boxplot(
    x='lang',
    y='delta_final_hybrid_score_100',
    data=merged,
    palette="Set2",
    ax=axes[0]
)

# Overlay mean and median
for i, lang in enumerate(merged['lang'].unique()):
    delta = merged[merged['lang'] == lang]['delta_final_hybrid_score_100']
    axes[0].scatter(i, delta.mean(), color='red', marker='o', s=60, label='Mean' if i==0 else "")
    axes[0].scatter(i, delta.median(), color='black', marker='D', s=40, label='Median' if i==0 else "")

    # Annotate sign effect size
    n_pos = sum(delta>0)
    n_neg = sum(delta<0)
    sign_effect = n_pos / (n_pos + n_neg)
    axes[0].text(i, delta.max() + 3, f"Sign effect: {sign_effect:.2f}", ha='center', fontsize=10, fontweight='bold')

axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
axes[0].set_xlabel("Language")
axes[0].set_ylabel("Delta Final Hybrid Score (Qwen − Gemma)")
axes[0].set_title("Qwen vs. Gemma: Score Differences")
axes[0].legend(loc='upper left')

# --- Right: Stacked bar of contributions ---
contrib_data.plot(
    kind="bar",
    stacked=True,
    ax=axes[1],
    colormap='Pastel1'
)
axes[1].axhline(0, color='gray', linewidth=1)
axes[1].set_ylabel("Contribution to Qwen−Gemma Score Difference (%)")
axes[1].set_xlabel("Language")
axes[1].set_title("Contribution Breakdown by Component")
axes[1].legend(title="Components", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Win rates dataframe
win_rates = pd.DataFrame({
    'qwen_win': [57.4, 70.6, 90.5],
    'gemma_win': [39.6, 15.7, 4.2]
}, index=['EN', 'ES', 'ZH'])

# Plot grouped bar chart
languages = win_rates.index
x = np.arange(len(languages))  # the label locations
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(8,5))
bars1 = ax.bar(x - width/2, win_rates['qwen_win'], width, label='Qwen', color='#66c2a5')
bars2 = ax.bar(x + width/2, win_rates['gemma_win'], width, label='Gemma', color='#fc8d62')

# Add value labels on top of bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

# Labels, title, and legend
ax.set_ylabel('Win Rate (%)')
ax.set_xlabel('Language')
ax.set_title('Win Rates of Qwen vs. Gemma by Language')
ax.set_xticks(x)
ax.set_xticklabels(languages)
ax.set_ylim(0, 100)
ax.legend()

plt.tight_layout()
plt.show()

# ============================================
#Comparing Qwen Base and Final Models
# ============================================
import pandas as pd

# Load files
base_df = pd.read_csv("jokes_qwenbase_final_hybrid_scored.csv")
final_df = pd.read_csv("jokes_qwen_final_hybrid_scored.csv")

# Normalize columns
base_df = base_df.rename(columns={
    "word_pair": "word_pair",
    "lang": "lang",
    "joke": "joke"
})

final_df = final_df.rename(columns={
    "rare_words": "word_pair",
    "language": "lang",
    "joke_text": "joke"
})

# Normalize word_pair formatting
base_df["word_pair"] = base_df["word_pair"].str.upper().str.replace("-", " & ")
final_df["word_pair"] = final_df["word_pair"].str.upper().str.strip()
base_pairs = set(base_df["word_pair"].unique())
final_pairs = set(final_df["word_pair"].unique())

print("Base model unique word pairs:", len(base_pairs))
print("Final model unique word pairs:", len(final_pairs))
base_density = base_df.groupby("word_pair").size()
final_density = final_df.groupby("word_pair").size()

coverage_summary = pd.DataFrame({
    "base_jokes_per_pair_mean": [base_density.mean()],
    "base_jokes_per_pair_min": [base_density.min()],
    "base_jokes_per_pair_max": [base_density.max()],
    "final_jokes_per_pair_mean": [final_density.mean()],
    "final_jokes_per_pair_min": [final_density.min()],
    "final_jokes_per_pair_max": [final_density.max()],
})

coverage_summary
common_pairs = set(base_df['word_pair']) & set(final_df['word_pair'])
len(common_pairs)
base_agg = (
    base_df[base_df['word_pair'].isin(common_pairs)]
    .groupby('word_pair')['final_hybrid_score_100']
    .mean()
    .reset_index(name='base_score')
)

final_agg = (
    final_df[final_df['word_pair'].isin(common_pairs)]
    .groupby('word_pair')['final_hybrid_score_100']
    .mean()
    .reset_index(name='final_score')
)
comparison_df = base_agg.merge(
    final_agg,
    left_on='word_pair',
    right_on='word_pair',
    how='inner'
)
comparison_df['delta'] = (
    comparison_df['final_score'] - comparison_df['base_score']
)
from scipy.stats import wilcoxon

paired_df = comparison_df.dropna(subset=["base_score", "final_score"])

stat, p = wilcoxon(
    paired_df["final_score"],
    paired_df["base_score"],
    alternative="two-sided"
)

stat, p
improvement = paired_df["final_score"] - paired_df["base_score"]

summary = {
    "mean_delta": improvement.mean(),
    "median_delta": improvement.median(),
    "positive_pct": (improvement > 0).mean() * 100,
    "n_pairs": len(paired_df)
}

summary
from scipy.stats import binomtest

pos = (improvement > 0).sum()
neg = (improvement < 0).sum()

sign_test = binomtest(
    k=pos,
    n=pos + neg,
    p=0.5,
    alternative="two-sided"
)

pos, neg, sign_test.pvalue
import numpy as np
from scipy.stats import mannwhitneyu

# Drop missing scores
base_scores = base_df["final_hybrid_score_100"].dropna().values
final_scores = final_df["final_hybrid_score_100"].dropna().values

len(base_scores), len(final_scores)
u_stat, p_value = mannwhitneyu(
    final_scores,
    base_scores,
    alternative="two-sided"
)

u_stat, p_value
def cliffs_delta(x, y):
    """
    Computes Cliff's Delta effect size.
    Returns value in [-1, 1]
    """
    x = np.asarray(x)
    y = np.asarray(y)

    greater = 0
    less = 0

    for xi in x:
        greater += np.sum(xi > y)
        less += np.sum(xi < y)

    delta = (greater - less) / (len(x) * len(y))
    return delta

delta = cliffs_delta(final_scores, base_scores)
delta


# ============================================
#Comparing Gemma Base and Final Models
# ============================================
#!pip install pingouin
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, norm
import pingouin as pg
base_df = pd.read_csv("jokes_gemma_base_final_hybrid_scored.csv")
final_df = pd.read_csv("jokes_gemma_final_hybrid_scored.csv")
print(len(base_df), len(final_df))
import re

def normalize_word_pair(wp):
    if pd.isna(wp):
        return None

    wp = wp.lower().strip()

    # Replace common separators with underscore
    wp = re.sub(r"\s*&\s*|\s*,\s*|\s+and\s+|\s+", "_", wp)

    parts = [p for p in wp.split("_") if p]

    # Keep only first two words (safety)
    parts = parts[:2]

    # Sort to make order-independent
    parts = sorted(parts)

    return "_".join(parts)
base_df["wp_norm"] = base_df["word_pair"].apply(normalize_word_pair)
final_df["wp_norm"] = final_df["word_pair"].apply(normalize_word_pair)
print(base_df["wp_norm"].unique()[:5])
print(final_df["wp_norm"].unique()[:5])
base_agg = (
    base_df
    .groupby("wp_norm", as_index=False)
    .agg(
        base_score=("final_hybrid_score_100", "mean"),
        overall_humor_base=("overall_humor", "mean"),
        rubric_score_total_base=("rubric_score_total", "mean")
    )
)
final_agg = (
    final_df
    .groupby("wp_norm", as_index=False)
    .agg(
        final_score=("final_hybrid_score_100", "mean"),
        overall_humor_final=("overall_humor", "mean"),
        rubric_score_total_final=("rubric_score_total", "mean")
    )
)
df = pd.merge(base_agg, final_agg, on="wp_norm", how="inner")

print(f"✅ Comparable word_pairs: {len(df)}")
df.head()
from scipy.stats import wilcoxon

stat, p = wilcoxon(df["final_score"], df["base_score"])

print("Wilcoxon signed-rank test")
print("Statistic:", stat)
print("p-value:", p)
import numpy as np

delta = df["final_score"] - df["base_score"]

print("Mean improvement:", delta.mean())
print("Median improvement:", delta.median())
print("Positive improvements (%):", (delta > 0).mean() * 100)
import numpy as np

r = (df["final_score"] - df["base_score"]).mean() / df["base_score"].std()
print("Effect size (approx):", r)
import matplotlib.pyplot as plt

plt.figure()
plt.boxplot([df["base_score"], df["final_score"]], labels=["Base", "Final"])
plt.ylabel("Final Hybrid Score")
plt.title("Gemma Base vs Final Model Performance")
plt.show()
import numpy as np
from scipy.stats import binomtest

delta = df["final_score"] - df["base_score"]

pos = (delta > 0).sum()
neg = (delta < 0).sum()

result = binomtest(pos, pos + neg, p=0.5, alternative="greater")

print("Positive:", pos)
print("Negative:", neg)
print("Sign test p-value:", result.pvalue)
