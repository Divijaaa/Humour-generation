import os
import json
import random
import subprocess
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# --- 1. CONFIGURATION & PATHS ---
SOURCE_FILE = "/Users/mohammadfaiz/Desktop/HPC/chinese_puns.jsonl"
TARGET_DIR = "/Users/mohammadfaiz/Desktop/HPC/chinese_puns_data"
ADAPTER_PATH_ = "./adapters_gemma_70k_puns_v2"
MODEL_NAME = "mlx-community/gemma-2-2b-4bit"

os.makedirs(TARGET_DIR, exist_ok=True)

# --- 2. DATA PREPARATION ---
formatted_data = []

if not os.path.exists(SOURCE_FILE):
    print(f"❌ Error: {SOURCE_FILE} not found.")
else:
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                w1 = str(data.get("word1", "")).strip()
                w2 = str(data.get("word2", "")).strip()
                joke_text = str(data.get("joke", "")).strip()

                if not joke_text: continue

                # Gemma-2 Chat Format (Consistent between Train and Test)
                # We use a simple prompt to focus the model on the puns
                prompt_content = f"关键词: {w1}, {w2}"
                text = f"<start_of_turn>user\n{prompt_content}<end_of_turn>\n<start_of_turn>model\n{joke_text}<end_of_turn>"
                
                formatted_data.append(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                continue

    # Shuffle and Split (90% Train, 10% Val)
    random.seed(3407)
    random.shuffle(formatted_data)
    split_idx = int(len(formatted_data) * 0.9)
    train_data = formatted_data[:split_idx]
    valid_data = formatted_data[split_idx:]

    with open(os.path.join(TARGET_DIR, "train.jsonl"), 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    with open(os.path.join(TARGET_DIR, "valid.jsonl"), 'w', encoding='utf-8') as f:
        f.writelines(valid_data)

    print(f"✅ Data Ready! Train: {len(train_data)} | Valid: {len(valid_data)}")

# --- 3. TRAINING COMMAND OUTPUT ---
print("\n--- COPY AND PASTE THIS INTO YOUR TERMINAL TO TRAIN ---")
print("python -m mlx_lm.lora \\")
print(f"  --model {MODEL_NAME} \\")
print(f"  --data {TARGET_DIR} \\")
print("  --train \\")
print("  --iters 35000 \\")
print("  --batch-size 2 \\")
print("  --grad-checkpoint \\")
print("  --learning-rate 1e-5 \\")
print("  --rank 8 \\")
print("  --alpha 20 \\")
print("  --steps-per-eval 500 \\")
print(f"  --adapter-path {ADAPTER_PATH}")
print("------------------------------------------------------\n")

# --- 4. INFERENCE FUNCTION ---
def ask_for_joke(word1, word2, model_obj, tokenizer_obj):
    # Matches the training prompt exactly
    prompt_content = f"关键词: {word1}, {word2}"
    full_prompt = f"<start_of_turn>user\n{prompt_content}<end_of_turn>\n<start_of_turn>model\n"

    sampler = make_sampler(temp=0.9)
    # Repetition penalty 1.2 is strong for Chinese to prevent character loops
    logits_processors = make_logits_processors(repetition_penalty=1.2)

    response = generate(
        model_obj, 
        tokenizer_obj, 
        prompt=full_prompt, 
        max_tokens=200,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False
    )
    return response

# --- 5. EXECUTION ---
if __name__ == "__main__":
    # Note: Only run this part AFTER you have trained and have adapters in ADAPTER_PATH_V1
    try:
        print("Loading model for testing...")
        model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH_V1)
        
        w1 = input("Enter Chinese Word 1: ")
        w2 = input("Enter Chinese Word 2: ")
        
        joke = ask_for_joke(w1, w2, model, tokenizer)
        print(f"\n✨ Generated Pun:\n{joke}")
    except Exception as e:
        print(f"\nSkipping test run: {e}")
        print("Tip: You need to complete a training run first to generate the adapter files.")
