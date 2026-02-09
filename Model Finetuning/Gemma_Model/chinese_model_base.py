import pandas as pd
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from tqdm import tqdm

# 1. Load the BASE model 
model_path = "mlx-community/gemma-2-2b-4bit" 

print("🔄 Loading Base Chinese Gemma-2-2B Model (Completion)...")
model, tokenizer = load(model_path)

# 2. Setup generation settings
sampler = make_sampler(temp=0.9)
# Base models need a higher repetition penalty to avoid getting stuck
logits_processors = make_logits_processors(repetition_penalty=1.2)

def generate_four_jokes(row):
    w1, w2 = row['word1'], row['word2']
    
    # FOR BASE MODELS: Provide a few-shot example pattern in Chinese
    # This teaches the model the format: Keywords -> Joke
    prompt = (
        f"关键词1: 手机\n关键词2: 充电器\n笑话: 手机和充电器吵架了，手机说：'你离我远点！' 充电器说：'好啊，等你有本事别来求我‘电’你！'\n"
        f"---\n"
        f"关键词1: {w1}\n关键词2: {w2}\n笑话:"
    )

    jokes = []
    for _ in range(4):
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=150,
            sampler=sampler,
            logits_processors=logits_processors
        )
        
        # Manually stop the generation if it tries to start a new pattern
        clean_joke = response.strip()
        for stop_word in ["---", "\n关键词", "关键词1:"]:
            if stop_word in clean_joke:
                clean_joke = clean_joke.split(stop_word)[0].strip()
        
        jokes.append(clean_joke)
    
    return pd.Series(jokes)

# 3. Load your JSONL file and LIMIT TO FIRST 50
input_file = '/Users/mohammadfaiz/Desktop/HPC/rare_combos.jsonl'
# Adjusting to .head(50) as requested
df = pd.read_json(input_file, lines=True).head(50)

print(f"🎤 Starting base model generation for {len(df)} pairs...")

# 4. Apply the function
tqdm.pandas()
df[['joke_1', 'joke_2', 'joke_3', 'joke_4']] = df.progress_apply(generate_four_jokes, axis=1)

# 5. Save to CSV
output_file = '/Users/mohammadfaiz/Desktop/HPC/Base_Model_chinese_rare_combos_50_rows_4_jokes.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"✅ Success! 200 Base Model Chinese jokes saved to: {output_file}")
