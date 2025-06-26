from transformers import pipeline
import math
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, EsmForProteinFolding
import numpy as np
import pandas as pd
import argparse
import yaml
import os
import shutil
import gc
from tqdm import tqdm

def create_or_replace_directory(directory_path):
    """
    Check if a directory exists. If it exists, delete it and create a new one.
    If it doesn't exist, create the directory.

    :param directory_path: The path to the directory
    """
    if os.path.exists(directory_path):
        print(f"Directory {directory_path} exists. Deleting...")
        shutil.rmtree(directory_path)
        print(f"Creating directory {directory_path}...")
        os.makedirs(directory_path)
    else:
        print(f"Directory {directory_path} does not exist. Creating...")
        os.makedirs(directory_path)

# === Arguments ===
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_path',            type=str, help='Path of the model to run generation from')
parser.add_argument('--num_return_sequences',  type=int, help='Number of sequences to generate', default=1000)
parser.add_argument('--max_length',            type=int, help='Maximum length of generated sequences', default=50)
parser.add_argument('--starts_with',           type=str, help='Starting amino acids for generation', default='')
parser.add_argument('--output_dir',            type=str, help='Directory for storing all output files')
parser.add_argument('--pred_model_path',       type=str, help='Path of the model to predict properties of the sequences')
parser.add_argument('--seed',                  type=int, help='Random seed', default=42)
parser.add_argument('--max_aa_count',          type=int, help='(for annotation only)', default=25)  # 🟢 EDITED
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Create output directory
create_or_replace_directory(args.output_dir)
protgpt2 = pipeline('text-generation', model=args.model_path, device_map="auto")

print('Starting generation...')
sequences = []
all_amines = ['L','A','G','V','E','S','I','K','R','D','T','P','N','Q','F','Y','M','H','C','W']
for cha in all_amines:
    batch = protgpt2(
        "<|endoftext|>" + cha,
        max_length=args.max_length,
        do_sample=True,
        top_k=950,
        repetition_penalty=1.2,
        num_return_sequences=args.num_return_sequences // len(all_amines),
        eos_token_id=0
    )
    sequences.extend(batch)
print('Generation Complete.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using Device', device)

# Tokenizer & model for perplexity
tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)

def calculatePerplexity(sequence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return math.exp(outputs.loss)

print('Starting to order by perplexity...')
ppls = []
sequences_with_ppl = []

for entry in tqdm(sequences):
    seq = entry['generated_text'].replace("<|endoftext|>", "").replace('\n', '').strip()
    sequences_with_ppl.append(seq)
    formatted = '<|endoftext|>' + '\n'.join(seq[i:i+60] for i in range(0, len(seq), 60)) + '<|endoftext|>'
    ppls.append(calculatePerplexity(formatted, model, tokenizer))

# Storing all generated proteins with their perplexity values in CSV
df_ppl = pd.DataFrame({
    'Sequence': df_seq := sequences_with_ppl,
    'Perplexity': df_ppl_vals := ppls
})
df_ppl['AA_Count'] = df_ppl['Sequence'].apply(len)  # 🟢 EDITED: annotate AA count
df_ppl.to_csv(os.path.join(args.output_dir, 'all_generated_with_perplexity.csv'), index=False)

k = args.num_return_sequences // 3
top_prots = np.array(sequences_with_ppl)[np.argsort(ppls)[:k]]
top_prots = [i for i in top_prots if len(i) != 0]
print('Ordered by Perplexity.')

# Checking if sequences are inside the hull
data = np.load('./hull_equations.npz')
def create_3d_point(sequence, c):
    for norm_char in ['X','U','B','Z','O','J']:
        sequence = sequence.replace(norm_char, '')
    count_c = sequence.count(c)
    if count_c == 0:
        return [0, 0, 0]
    indices_c = [i for i, char in enumerate(sequence) if char == c]
    mean_distance = sum(indices_c) / count_c
    variance = sum((idx - mean_distance) ** 2 for idx in indices_c) / count_c
    second_moment = variance / len(sequence)
    return [count_c, mean_distance, second_moment]

def point_in_hull(point, hull_equations, tolerance=1e-12):
    return all(np.dot(eq[:-1], point) + eq[-1] <= tolerance for eq in hull_equations)

def check_validity(seq):
    return all(point_in_hull(create_3d_point(seq, c), data[c]) for c in data.files)

print('Starting Protein Validity Check...')
df_valid = pd.DataFrame({'Sequence': top_prots})
df_valid['Valid Protein'] = df_valid['Sequence'].apply(check_validity)
filtered_prots = [prot for prot in top_prots if check_validity(prot)]
print('Valid proteins identified.')

# Structure check
del model, protgpt2
gc.collect()
torch.cuda.empty_cache()
print('Starting structure check...')
esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

plddt_results = []
for sequence in tqdm(filtered_prots):
    inputs = esm_tokenizer([sequence], return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = esm_model(**inputs)
    plddt_results.append(torch.mean(outputs.plddt).item())
df_valid['plDDT'] = pldt_results
df_valid['Good Structure'] = df_valid['plDDT'] > 0.7
df_valid.to_csv(os.path.join(args.output_dir, 'generation_checks.csv'), index=False)
best_prots = df_valid[df_valid['Good Structure']]['Sequence'].tolist()
print('Structure check complete.')

# Loading PeptideBERT model
from PeptideBERT.data.dataloader import load_data
from PeptideBERT.model.network import create_model, cri_opt_sch
from PeptideBERT.model.utils import train, validate, test

config = yaml.safe_load(open(os.path.join(args.pred_model_path, 'config.yaml'), Loader=yaml.FullLoader))
config['device'] = device
peptideBERT_model = create_model(config)
peptideBERT_model.load_state_dict(
    torch.load(os.path.join(args.pred_model_path, 'model.pt'))['model_state_dict'], strict=False)

def f(seq):
    m2 = dict(zip(
        ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]'] + all_amines + ['X','U','B','Z','O'],
        range(30)
    ))
    tensor = torch.tensor([m2[c] for c in seq], dtype=torch.long).unsqueeze(0).to(device)
    attn = (tensor > 0).long().to(device)
    return tensor, attn

print('Starting protein property predictions...')
scores = {}
for seq in tqdm(best_prots):
    scores[seq] = peptideBERT_model(*f(seq)).item()

df_preds = pd.DataFrame({
    'Sequence': list(scores.keys()),
    'Score': list(scores.values())
})
df_preds['Property'] = df_preds['Score'] > 0.5
df_preds.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
print('Property prediction completed.')

with open(os.path.join(args.output_dir, 'info.txt'), 'w') as file:
    file.write(f'Total generated sequences: {args.num_return_sequences}\n')
    file.write(f'Top by perplexity: {len(top_prots)}\n')
    file.write(f'Passed validity check: {len(filtered_prots)}\n')
    file.write(f'Passed plDDT > 0.7: {len(best_prots)}\n')
    file.write(f'Property-positive: {sum(df_preds["Property"])}/{len(best_prots)}\n')
print('Inference Complete')

