import os
import json
import pandas as pd
import subprocess
from sklearn.model_selection import train_test_split
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# --- 1. CONFIGURATION & PATHS ---
INPUT_FILE = '/Users/mohammadfaiz/Desktop/HPC/shortjokes_en_pairs.jsonl'
OUTPUT_DIR = '/Users/mohammadfaiz/Desktop/HPC/english_puns_data'
ADAPTER_PATH = "./adapters_gemma_english_puns"
MODEL_NAME = "mlx-community/gemma-2-2b-4bit"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. DATA PREPARATION ---
if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: {INPUT_FILE} not found.")
else:
    df = pd.read_json(INPUT_FILE, lines=True)

    def format_gemma_text(row):
        # We include the persona here so the model associates this style with the instruction
        persona = "You are a witty English comedian. Generate a funny joke containing the two given words."
        keywords = f"Keywords: {row['word1']}, {row['word2']}"
        joke = row['joke']
        
        # Clean Gemma-2 Template (No manual <bos>/<eos> - MLX adds these)
        return f"<start_of_turn>user\n{persona}\n{keywords}<end_of_turn>\n<start_of_turn>model\n{joke}<end_of_turn>"

    df['text'] = df.apply(format_gemma_text, axis=1)

    # Split: 80% Train, 10% Valid, 10% Test
    train_df, temp_df = train_test_split(df, test_size=0.1, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    def save_jsonl(dataframe, filename):
        with open(os.path.join(OUTPUT_DIR, filename), 'w', encoding='utf-8') as f:
            for text in dataframe['text']:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

    save_jsonl(train_df, "train.jsonl")
    save_jsonl(valid_df, "valid.jsonl")
    save_jsonl(test_df, "test.jsonl")

    print(f"✅ English Data Ready! Train: {len(train_df)}, Valid: {len(valid_df)}")

# --- 3. TRAINING COMMAND ---
print("\n--- RUN THIS IN TERMINAL ---")
print("python -m mlx_lm.lora \\")
print(f"  --model {MODEL_NAME} \\")
print(f"  --data {OUTPUT_DIR} \\")
print("  --train \\")
print("  --iters 55000 \\")
print("  --batch-size 1 \\")
print("  --learning-rate 1e-5 \\")
print("  --rank 8 \\")
print("  --alpha 20 \\")
print("  --grad-checkpoint \\")
print(f"  --adapter-path {ADAPTER_PATH}")
print("----------------------------\n")

# --- 4. INFERENCE FUNCTION ---
def ask_for_joke(word1, word2, model_obj, tokenizer_obj):
    # Matches the persona and format used in training EXACTLY
    persona = "You are a witty English comedian. Generate a funny joke containing the two given words."
    keywords = f"Keywords: {word1}, {word2}"
    
    full_prompt = f"<start_of_turn>user\n{persona}\n{keywords}<end_of_turn>\n<start_of_turn>model\n"

    sampler = make_sampler(temp=0.8)
    logits_processors = make_logits_processors(repetition_penalty=1.2)

    print(f"🎤 [Generating English joke for: {word1} + {word2}...]")

    response = generate(
        model_obj, 
        tokenizer_obj, 
        prompt=full_prompt, 
        max_tokens=200,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=True
    )
    return response

# --- 5. TEST BLOCK ---
if __name__ == "__main__":
    try:
        model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH)
        w1 = input("Enter Word 1: ")
        w2 = input("Enter Word 2: ")
        ask_for_joke(w1, w2, model, tokenizer)
    except Exception as e:
        print("\nModel/Adapters not loaded yet. Train first!")
