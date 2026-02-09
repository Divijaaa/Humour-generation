import os
import json
import pandas as pd
import subprocess

# 1. INSTALL LIBRARIES
print("--- Installing Mac-specific AI libraries ---")
subprocess.run(["pip", "install", "-U", "mlx-lm", "pandas"])

# --- 2. DATA PREPARATION ---
input_file = "spanish_joke_triplets.jsonl"

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
else:
    df = pd.read_json(input_file, lines=True)
    
    # Shuffle only (no downsampling)
    df = df.sample(frac=1, random_state=3407).reset_index(drop=True)
    
    # Split: 90% Training, 10% Validation
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx]
    valid_df = df.iloc[split_idx:]

    print(f"--- Preparing {len(train_df)} training and {len(valid_df)} validation samples ---")

    def format_entry(row):
        # Note: Using Gemma-2's specific control tokens <start_of_turn> 
        # often yields better results than Alpaca headers for this specific model.
        prompt = (f"<start_of_turn>user\nEscribe un chiste en español que contenga las siguientes palabras: "
                  f"{row['word1']}, {row['word2']}<end_of_turn>\n"
                  f"<start_of_turn>model\n{row['joke']}<end_of_turn>")
        return {"text": prompt}

    os.makedirs("data", exist_ok=True)
    with open("data/train.jsonl", "w", encoding="utf-8") as f:
        for _, row in train_df.iterrows():
            f.write(json.dumps(format_entry(row), ensure_ascii=False) + "\n")
    
    with open("data/valid.jsonl", "w", encoding="utf-8") as f:
        for _, row in valid_df.iterrows():
            f.write(json.dumps(format_entry(row), ensure_ascii=False) + "\n")

    print("Success: Full dataset saved to ./data/")

# --- 3. TRAINING INSTRUCTIONS ---

print("READY FOR TRAINING ON YOUR MAC GPU")
print("Run the following command in your TERMINAL to start training:")
print("\npython -m mlx_lm.lora \\")
print("  --model mlx-community/gemma-2-2b-4bit \\")
print("  --data ./data \\")
print("  --train \\")
print("  --iters 17500 \\")
print("  --batch-size 1 \\")
print("  --learning-rate 2e-5 \\")
print("  --rank 8 \\")
print("  --alpha 20 \\")
print("  --adapter-path ./adapters_gemma_v7")

from mlx_lm import load, generate, sample_utils

# 1. Setup Paths
model_path = "mlx-community/gemma-2-2b-4bit"
adapter_path = "adapters_gemma_v7"  # Pointing to the new version

# 2. Load Model & Adapters
print("Loading the refined humor model...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

def generate_custom_joke(word1, word2):
    # Match the EXACT format used in format_entry()
    prompt = (f"<start_of_turn>user\nEscribe un chiste en español que contenga las siguientes palabras: "
              f"{word1}, {word2}<end_of_turn>\n"
              f"<start_of_turn>model\n")
    
    sampler = sample_utils.make_sampler(temp=0.7)
    if hasattr(sampler, "set_repetition_penalty"):
        sampler.set_repetition_penalty(1.1, 64)
    
    print(f"--- Generating joke with: {word1} & {word2} ---")
    
    output = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=200,
        sampler=sampler,
        verbose=True
    )
    return output

# --- TEST IT ---
# Try words that are NOT common in Jaimito jokes to see if it follows instructions
result = generate_custom_joke("computadora", "frío")


