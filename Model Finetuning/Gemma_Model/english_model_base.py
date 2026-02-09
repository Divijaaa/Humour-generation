import pandas as pd
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# 1. Load the BASE model
model_path = "mlx-community/gemma-2-2b-4bit" 

print("🔄 Loading Base Gemma-2-2B Model...")
model, tokenizer = load(model_path)

# Setup sampler (High temp for variety)
sampler = make_sampler(temp=0.9)

# Fix: make_logits_processors ONLY takes repetition parameters
lps = make_logits_processors(repetition_penalty=1.2)

def generate_four_jokes(word1, word2):
    jokes = []
    
    # Prompt for Base Model pattern matching
    prompt = (
        f"Keyword 1: hammer\nKeyword 2: flower\nJoke: Why did the hammer bring a flower to work? Because it heard there was a smash-hit bouquet!\n"
        f"---\n"
        f"Keyword 1: {word1}\nKeyword 2: {word2}\nJoke:"
    )

    for i in range(4):
        # Generate full completion
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=100, 
            sampler=sampler,
            logits_processors=lps
        )
        
        # --- STOP SEQUENCE WORKAROUND ---
        # Since 'generate' doesn't support stop_sequences in this version,
        # we manually split the text at our markers.
        clean_joke = response.strip()
        
        # Stop at any of these markers if the model tries to generate a new row
        for stop_word in ["---", "\nKeyword", "Keyword 1:", "Keyword 2:"]:
            if stop_word in clean_joke:
                clean_joke = clean_joke.split(stop_word)[0].strip()
        
        jokes.append(clean_joke)
    
    return pd.Series(jokes)

# 2. Load the input CSV
input_path = "/Users/mohammadfaiz/Desktop/HPC/eng_rarewords.csv"
df = pd.read_csv(input_path).head(100)

print(f"🎤 Processing {len(df)} pairs with Base Model...")

# 3. Process each row
df[['joke_1', 'joke_2', 'joke_3', 'joke_4']] = df.apply(
    lambda row: generate_four_jokes(row['word_1'], row['word_2']), axis=1
)

# 4. Save results
output_path = "/Users/mohammadfaiz/Desktop/HPC/Base_Model_eng_rarewords_4_jokes.csv"
df.to_csv(output_path, index=False)

print(f"✅ Finished! Saved to: {output_path}")
