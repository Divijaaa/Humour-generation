import pandas as pd
from tqdm import tqdm
from mlx_lm import load, generate, sample_utils

# --- SETUP ---
# 1. Load the BASE model (Ensure it is NOT the -it version)
model_path = "mlx-community/gemma-2-2b-4bit"

print("🔄 Cargando el modelo Base Gemma-2-2B (Texto Completado)...")
model, tokenizer = load(model_path)

# 2. Setup generation settings
# One sampler for all, temp 0.9 for variety among the 4 jokes
sampler = sample_utils.make_sampler(temp=0.9)
# Repetition penalty helps the base model avoid getting stuck in loops
lps = sample_utils.make_logits_processors(repetition_penalty=1.2)

def generate_single_joke(word1, word2):
    """
    Generates a joke using pattern completion (one attempt).
    """
    # FEW-SHOT PROMPT: We give the model one example to follow.
    # Base models work best when they see a pattern they can copy.
    prompt = (
        f"Palabra 1: martillo\nPalabra 2: flor\n"
        f"Chiste: ¿Por qué el martillo llevó una flor al trabajo? ¡Porque quería dar un golpe de aroma!\n"
        f"---\n"
        f"Palabra 1: {word1}\nPalabra 2: {word2}\n"
        f"Chiste:"
    )

    # Generate the completion
    output = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=100, 
        sampler=sampler,
        logits_processors=lps
    )
    
    # Clean the output
    # We split by '---' or '\nPalabra' to stop the model if it tries to hallucinate a new row
    clean_joke = output.strip().split("---")[0].split("\nPalabra")[0].strip()
    
    return clean_joke

# --- CSV PROCESSING ---

# 1. Load the input CSV and LIMIT TO FIRST 50
df = pd.read_csv('rare_combos.csv').head(50)

print(f"🚀 Procesando las primeras {len(df)} filas. 1 intento por chiste (200 total)...")

# 2. Storage for results
joke1_results, joke2_results, joke3_results, joke4_results = [], [], [], []

# 3. Loop through the rows
for index, row in tqdm(df.iterrows(), total=len(df)):
    # Using 'verb' and 'noun' from your original Spanish dataset
    w1, w2 = row['verb'], row['noun']
    
    # Just one attempt per joke column
    joke1_results.append(generate_single_joke(w1, w2))
    joke2_results.append(generate_single_joke(w1, w2))
    joke3_results.append(generate_single_joke(w1, w2))
    joke4_results.append(generate_single_joke(w1, w2))

# 4. Add the results as new columns
df['chiste_1'] = joke1_results
df['chiste_2'] = joke2_results
df['chiste_3'] = joke3_results
df['chiste_4'] = joke4_results

# 5. Save to a new CSV file
output_file = 'Base_Model_spanish_rare_combos_50rows_4jokes.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n--- ¡HECHO! ---")
print(f"Resultados del modelo BASE guardados en '{output_file}'")
