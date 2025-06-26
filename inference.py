# inference.py
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
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_return_sequences', type=int, default=1000)
parser.add_argument('--max_length', type=int, default=50)
parser.add_argument('--starts_with', type=str, default='')
parser.add_argument('--output_dir', type=str)
parser.add_argument('--pred_model_path', type=str)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_aa_count', type=int, default=25)  # ✅ Added
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

create_or_replace_directory(args.output_dir)
protgpt2 = pipeline('text-generation', model=args.model_path, device_map="auto")

print('🔄 Generating sequences...')
sequences = []
all_amines = ['L','A','G','V','E','S','I','K','R','D','T','P','N','Q','F','Y','M','H','C','W']
for cha in all_amines:
    batch = protgpt2("<|endoftext|>" + cha, max_length=args.max_length, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=args.num_return_sequences // len(all_amines), eos_token_id=0)
    sequences.extend(batch)
print(f"✅ Generated {len(sequences)} raw sequences.")

# ✅ Filter by amino acid count BEFORE perplexity
filtered_sequences = []
for entry in sequences:
    text = entry['generated_text'].replace("<|endoftext|>", "").strip().replace("\n", "")
    if len(text) <= args.max_aa_count:
        filtered_sequences.append(text)
print(f"✅ Retained {len(filtered_sequences)} sequences after filtering by max_aa_count = {args.max_aa_count}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)

def calculatePerplexity(sequence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return math.exp(loss)

print('⚙️ Calculating perplexity...')
ppls, sequences_with_ppl = [], []
for seq in tqdm(filtered_sequences):
    formatted = '<|endoftext|>' + seq + '<|endoftext|>'
    sequences_with_ppl.append(seq)
    ppl = calculatePerplexity(formatted, model, tokenizer)
    ppls.append(ppl)

df_ppl = pd.DataFrame({'Sequence': sequences_with_ppl, 'Perplexity': ppls})
df_ppl.to_csv(os.path.join(args.output_dir, 'all_generated_with_perplexity.csv'), index=False)

k = args.num_return_sequences // 3
top_prots = np.array(sequences_with_ppl)[np.argsort(ppls)[:k]]
top_prots = [seq for seq in top_prots if len(seq) > 0]

print(f"✅ Selected top {len(top_prots)} sequences by perplexity.")

# Hull filtering
data = np.load('./hull_equations.npz')
def create_3d_point(seq, c):
    for ch in ['X','U','B','Z','O','J']:
        seq = seq.replace(ch, '')
    count = seq.count(c)
    if count == 0:
        return [0, 0, 0]
    idxs = [i for i, char in enumerate(seq) if char == c]
    mean = sum(idxs)/count
    var = sum((i - mean)**2 for i in idxs)/count
    return [count, mean, var/len(seq)]

def point_in_hull(point, equations, tol=1e-12):
    return all(np.dot(eq[:-1], point) + eq[-1] <= tol for eq in equations)

def check_validity(seq):
    return all(point_in_hull(create_3d_point(seq, c), data[c]) for c in data.files)

df_valid = pd.DataFrame({'Sequence': top_prots})
df_valid['Valid Protein'] = df_valid['Sequence'].apply(check_validity)
filtered_prots = df_valid[df_valid['Valid Protein']]['Sequence'].tolist()
print(f"✅ {len(filtered_prots)} proteins passed hull validity check.")

# ESMFold Structure
esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

plddt_results = []
for seq in tqdm(filtered_prots):
    clean_seq = seq.replace('\n', '')
    inputs = esm_tokenizer([clean_seq], return_tensors="pt", padding=True, truncation=True).to(device)  # ✅ Fixed crash
    outputs = esm_model(**inputs)
    plddt_results.append(torch.mean(outputs.plddt).item())

df_valid = df_valid[df_valid['Valid Protein']]
df_valid['plDDT'] = plddt_results
df_valid['Good Structure'] = df_valid['plDDT'] > 0.7
df_valid.to_csv(os.path.join(args.output_dir, 'generation_checks.csv'), index=False)

best_prots = df_valid[df_valid['Good Structure']]['Sequence'].tolist()

# PeptideBERT
from PeptideBERT.data.dataloader import load_data
from PeptideBERT.model.network import create_model
from PeptideBERT.model.utils import train, validate, test

config = yaml.safe_load(open(os.path.join(args.pred_model_path, 'config.yaml')))
config['device'] = device
peptideBERT_model = create_model(config)
peptideBERT_model.load_state_dict(torch.load(os.path.join(args.pred_model_path, 'model.pt'))['model_state_dict'], strict=False)

m2 = dict(zip(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','L','A','G','V','E','S','I','K','R','D','T','P','N','Q','F','Y','M','H','C','W','X','U','B','Z','O'], range(30)))
def f(seq):
    tensor = torch.tensor([m2[c] for c in seq], dtype=torch.long).unsqueeze(0).to(device)
    attn = torch.tensor(tensor > 0, dtype=torch.long).to(device)
    return tensor, attn

scores = {}
print('🔍 Predicting property with PeptideBERT...')
for seq in tqdm(best_prots):
    scores[seq] = peptideBERT_model(*f(seq)).item()

df_preds = pd.DataFrame({'Sequence': list(scores.keys()), 'Score': list(scores.values())})
df_preds['Property'] = df_preds['Score'] > 0.5
df_preds.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

# Info
with open(os.path.join(args.output_dir, 'info.txt'), 'w') as f:
    f.write(f"Total generated sequences: {args.num_return_sequences}\n")
    f.write(f"Filtered by max_aa_count ≤ {args.max_aa_count}: {len(filtered_sequences)}\n")
    f.write(f"Top by perplexity: {k}\n")
    f.write(f"Passed validity check: {len(filtered_prots)}\n")
    f.write(f"Passed plDDT > 0.7: {len(best_prots)}\n")
    f.write(f"Property-positive: {sum(df_preds['Property'])}/{len(df_preds)}\n")

print("✅ Inference completed.")
