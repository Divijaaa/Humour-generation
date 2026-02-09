now similarly help me clean my code for the chinese model
import json
import os
import random

# Paths
source_file = "/Users/mohammadfaiz/Desktop/HPC/chinese_puns.jsonl"
target_dir = "/Users/mohammadfaiz/Desktop/HPC/chinese_puns_data"
os.makedirs(target_dir, exist_ok=True)

formatted_data = []

with open(source_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: 
            continue
            
        try:
            data = json.loads(line)
            
            # 1. Extract your specific columns
            w1 = str(data.get("word1", "")).strip()
            w2 = str(data.get("word2", "")).strip()
            joke_text = str(data.get("joke", "")).strip()
            
            # 2. Skip if the joke is empty
            if not joke_text:
                continue

            # 3. SPEED HACK: Keep response length manageable
            # Capping at 200 chars keeps the 'it/sec' high on Apple Silicon
            if len(joke_text) > 200:
                joke_text = joke_text[:200] + "..."
            
            # 4. Create the formatted prompt
            # Keeping the user prompt short also speeds up training
            prompt = f"关键词: {w1}, {w2}"
            
            # Gemma-2 Chat Format
            text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{joke_text}<end_of_turn>"
            
            # Add to list as a JSON line
            formatted_data.append(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            
        except json.JSONDecodeError:
            continue

# Shuffle to mix up the patterns
random.shuffle(formatted_data)

# Splitting logic
train_count = 70000
valid_count = 2000

train_data = formatted_data[:train_count]
valid_data = formatted_data[train_count : train_count + valid_count]

# Save the files
with open(os.path.join(target_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
    f.writelines(train_data)
with open(os.path.join(target_dir, "valid.jsonl"), 'w', encoding='utf-8') as f:
    f.writelines(valid_data)

print(f"✅ Reformatting Complete!")
print(f"Train samples: {len(train_data)}")

!python -m mlx_lm.lora \
  --model mlx-community/gemma-2-2b-4bit \
  --data /Users/mohammadfaiz/Desktop/HPC/chinese_puns_v2 \
  --train \
  --resume-adapter-file ./adapters_gemma_70k_puns/adapters.safetensors \
  --iters 70000 \
  --batch-size 1 \
  --num-layers 8 \
  --grad-checkpoint \
  --learning-rate 1e-5 \
  --steps-per-eval 500 \
  --save-every 1000 \
  --adapter-path ./adapters_gemma_70k_puns_v2
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# 1. Load the model and adapters once
model_path = "mlx-community/gemma-2-2b-it-4bit"
adapter_path = "./adapters_gemma_70k_puns"
model, tokenizer = load(model_path, adapter_path=adapter_path)

def ask_for_joke(word1, word2):
    # Prompt Setup
    persona = "You are a Chinese comedian. Generate a very funny joke in Chinese containing the two given words."
    prompt_content = f"关键词: {word1}, {word2}"
    full_prompt = f"<start_of_turn>user\n{persona}\n{prompt_content}<end_of_turn>\n<start_of_turn>model\n"

    # 2. THE REPETITION FIX
    # make_sampler handles the randomness (temperature)
    sampler = make_sampler(temp=0.8)
    
    # make_logits_processors handles the repetition penalty
    # 1.2 is a strong penalty that breaks "Da Da Da" loops
    logits_processors = make_logits_processors(repetition_penalty=1.2)

    # 3. Generate
    response = generate(
        model, 
        tokenizer, 
        prompt=full_prompt, 
        max_tokens=200,
        sampler=sampler,
        logits_processors=logits_processors
    )
    print(f"\nResult:\n{response}")

# --- Custom Input ---
w1 = input("Enter Word 1: ")
w2 = input("Enter Word 2: ")
ask_for_joke(w1, w2)
