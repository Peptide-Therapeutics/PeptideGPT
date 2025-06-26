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
import stat
import gc
from tqdm import tqdm

def handle_remove_readonly(func, path, exc):
    """Allow shutil.rmtree to delete readonly files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

# ── Arguments ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='PeptideGPT inference with AA_Count annotation')
parser.add_argument('--model_path',           type=str, required=True, help='Path of the PeptideGPT model folder')
parser.add_argument('--num_return_sequences', type=int, default=1000, help='Number of sequences to generate')
parser.add_argument('--max_length',           type=int, default=50,   help='Maximum length of generated sequences')
parser.add_argument('--starts_with',          type=str, default='',   help='Starting amino acids for generation')
parser.add_argument('--output_dir',           type=str, required=True, help='Directory for storing all output files')
parser.add_argument('--pred_model_path',      type=str, required=True, help='Path of the PeptideBERT model folder')
parser.add_argument('--seed',                 type=int, default=42,   help='Random seed')
parser.add_argument('--max_aa_count',         type=int, default=25,   help='Annotation only (not used for filtering)')  # EDITED
args = parser.parse_args()

# ── Safe‐remove & recreate output directory ─────────────────────────────────────
if os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir, onerror=handle_remove_readonly)
os.makedirs(args.output_dir, exist_ok=True)

# ── Print model path & contents ────────────────────────────────────────────────
print("GPT model path:", os.path.abspath(args.model_path))
print("Model directory contents:", os.listdir(os.path.abspath(args.model_path)))

# ── Seeding ─────────────────────────────────────────────────────────────────────
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── Generation ─────────────────────────────────────────────────────────────────
print('🔄 Generating sequences...')
protgpt2 = pipeline('text-generation', model=args.model_path, device_map="auto")
all_amines = list("LAGVESIKRDTPNQFYMHCW")
sequences = []
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
print(f"✅ Generated {len(sequences)} raw sequences.")

# ── Prepare plain sequences ─────────────────────────────────────────────────────
sequence_texts = [
    entry['generated_text']
         .replace("<|endoftext|>", "")
         .strip()
         .replace("\n","")
    for entry in sequences
]

# ── Perplexity Scoring ─────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
model     = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)

def calculatePerplexity(seq):
    ids = torch.tensor(tokenizer.encode(seq)).unsqueeze(0).to(device)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return math.exp(loss)

print('⚙️ Calculating perplexity...')
ppls = []
for s in tqdm(sequence_texts):
    formatted = '<|endoftext|>' + '\n'.join(s[i:i+60] for i in range(0, len(s), 60)) + '<|endoftext|>'
    ppls.append(calculatePerplexity(formatted))

# ── Write Perplexity CSV (with AA_Count) ────────────────────────────────────────
df_ppl = pd.DataFrame({
    'Sequence':   sequence_texts,
    'Perplexity': ppls
})
# Annotate with amino‐acid count (no filtering)
df_ppl['AA_Count'] = df_ppl['Sequence'].str.len()
out_csv = os.path.join(args.output_dir, 'all_generated_with_perplexity_with_lengths.csv')
df_ppl.to_csv(out_csv, index=False)
print(f"✅ Saved perplexity + AA_Count: {out_csv}")

# ── Top-K Selection ─────────────────────────────────────────────────────────────
k = args.num_return_sequences // 3
top_prots = df_ppl.nsmallest(k, 'Perplexity')['Sequence'].tolist()
print(f"✅ Selected top {len(top_prots)} sequences by perplexity.")

# ── Hull Filtering ──────────────────────────────────────────────────────────────
data = np.load('./hull_equations.npz')
def create_3d_point(seq, c):
    for ch in "XUBZOJ":
        seq = seq.replace(ch, '')
    cnt = seq.count(c)
    if cnt == 0: return [0,0,0]
    idxs = [i for i,ch in enumerate(seq) if ch==c]
    mean = sum(idxs)/cnt
    var  = sum((i-mean)**2 for i in idxs)/cnt
    return [cnt, mean, var/len(seq)]

def point_in_hull(pt, eqs, tol=1e-12):
    return all(np.dot(e[:-1], pt) + e[-1] <= tol for e in eqs)

df_valid = pd.DataFrame({'Sequence': top_prots})
df_valid['ValidProtein'] = df_valid['Sequence'].apply(lambda s: all(point_in_hull(create_3d_point(s,c), data[c]) for c in data.files))
filtered_prots = df_valid[df_valid['ValidProtein']]['Sequence'].tolist()
print(f"✅ {len(filtered_prots)} proteins passed hull validity check.")

# ── ESMFold Structure ──────────────────────────────────────────────────────────
esm_tok   = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)

plddt_results = []
for seq in tqdm(filtered_prots):
    inp = esm_tok([seq], return_tensors="pt", padding=True, truncation=True).to(device)
    out = esm_model(**inp)
    plddt_results.append(torch.mean(out.plddt).item())

df_valid = df_valid[df_valid['ValidProtein']].copy()
df_valid['plDDT']         = plddt_results
df_valid['GoodStructure'] = df_valid['plDDT'] > 0.7
csv2 = os.path.join(args.output_dir, 'generation_checks.csv')
df_valid.to_csv(csv2, index=False)
best_prots = df_valid[df_valid['GoodStructure']]['Sequence'].tolist()
print(f"✅ {len(best_prots)} proteins passed plDDT > 0.7.")

# ── PeptideBERT Prediction ──────────────────────────────────────────────────────
from PeptideBERT.data.dataloader import load_data
from PeptideBERT.model.network import create_model
from PeptideBERT.model.utils    import train, validate, test

config = yaml.safe_load(open(os.path.join(args.pred_model_path, 'config.yaml'), Loader=yaml.FullLoader))
config['device'] = device
pB_model = create_model(config)
pB_model.load_state_dict(torch.load(os.path.join(args.pred_model_path, 'model.pt'))['model_state_dict'], strict=False)

m2 = dict(zip(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]'] + all_amines + ['X','U','B','Z','O'], range(30)))
def featurize(s):
    t = torch.tensor([m2[c] for c in s], dtype=torch.long).unsqueeze(0).to(device)
    a = (t>0).long().to(device)
    return t, a

scores = {}
print('🔍 Predicting property with PeptideBERT...')
for seq in tqdm(best_prots):
    scores[seq] = pB_model(*featurize(seq)).item()

df_preds = pd.DataFrame({'Sequence': list(scores), 'Score': list(scores.values())})
df_preds['Property'] = df_preds['Score'] > 0.5
csv3 = os.path.join(args.output_dir, 'predictions.csv')
df_preds.to_csv(csv3, index=False)
print('🔍 Property predictions saved.')

# ── Summary Info ───────────────────────────────────────────────────────────────
with open(os.path.join(args.output_dir, 'info.txt'), 'w') as f:
    f.write(f"Total generated sequences        : {len(sequence_texts)}\n")
    f.write(f"AA_Count annotation threshold   : {args.max_aa_count}\n")  # this is just for record
    f.write(f"Top by perplexity               : {len(top_prots)}\n")
    f.write(f"Passed hull validity            : {len(filtered_prots)}\n")
    f.write(f"Passed plDDT > 0.7              : {len(best_prots)}\n")
    f.write(f"Property-positive               : {sum(df_preds['Property'])}/{len(best_prots)}\n")
print("✅ Inference completed.")

