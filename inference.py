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
    
    # Check if the directory exists
    if os.path.exists(directory_path):
        print(f"Directory {directory_path} exists. Deleting...")
        
        # Delete the directory
        shutil.rmtree(directory_path)
        
        # Create the directory
        print(f"Creating directory {directory_path}...")
        os.makedirs(directory_path)
    else:
        # If the directory doesn't exist, create it
        print(f"Directory {directory_path} does not exist. Creating...")
        os.makedirs(directory_path)


# Specify the directory path


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_path', type=str, help='Path of the model to run generation from')
parser.add_argument('--num_return_sequences', type=int, help='Number of sequences to generate', default=1000)
parser.add_argument('--max_length', type=int, help='Maximum length of generated sequences', default=50)
parser.add_argument('--starts_with', type=str, help='Starting amino acids for generation', default='')
parser.add_argument('--output_dir', type=str, help='Directory for storing all output files')
parser.add_argument('--pred_model_path', type=str, help='Path of the model to predict properties of the sequences')
parser.add_argument('--seed', type=int, help='Random seed', default=42)



args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Create the directory
create_or_replace_directory(args.output_dir)
protgpt2 = pipeline('text-generation', model=args.model_path, device_map="auto")

print('Starting generation...')
sequences = []
all_amines = ['L','A','G','V','E','S','I','K','R','D','T','P','N','Q','F','Y','M','H','C','W']
for cha in all_amines:
    sequences_cha = protgpt2("<|endoftext|>" + cha, max_length=args.max_length, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=args.num_return_sequences//len(all_amines), eos_token_id=0)
    sequences.extend(sequences_cha)
print('Generation Complete.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using Device', device)

#Convert the sequence to a string like this
#(note we have to introduce new line characters every 60 amino acids,
#following the FASTA file format).
tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
model = GPT2LMHeadModel.from_pretrained(args.model_path)
model = model.to(device)

# ppl function
def calculatePerplexity(sequence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    return math.exp(loss)


print('Starting to order by perplexity...')

ppls = []
sequences_with_ppl = []

for sequence in tqdm(sequences):
    seq = sequence['generated_text']
    seq = seq.replace("<|endoftext|>", "")
    seq = seq.replace('\n', '')
    seq = seq.strip()
    sequences_with_ppl.append(seq)
    seq = '\n'.join(seq[i:i+60] for i in range(0, len(seq), 60))
    seq = '<|endoftext|>' + seq + '<|endoftext|>'
    ppl = calculatePerplexity(seq, model, tokenizer)
    ppls.append(ppl)

#storing all generated proteins with their perplexity values in csv
df_ppl = pd.DataFrame()
df_ppl['Sequence'] = sequences_with_ppl
df_ppl['Perplexity'] = ppls
df_ppl.to_csv(args.output_dir + '/all_generated_with_perplexity.csv', index=False)

k = args.num_return_sequences//3

top_prots = np.array(sequences_with_ppl)[np.argsort(ppls)[:k]]
top_prots = [i for i in top_prots if len(i) != 0]
print(top_prots)

def remove_prots_with(lst, chars_to_exclude):
    # Convert the exclusion string to a set for efficient lookup
    exclusion_set = set(chars_to_exclude)
    
    # Use a list comprehension to filter out strings
    filtered_list = [s for s in lst if not any(char in exclusion_set for char in s)]
    
    return filtered_list

# Example usage:




print('Ordered by Perplexity.')

#checking if sequences are inside the hull
# Load the .npz file
data = np.load('./hull_equations.npz')

df_valid = pd.DataFrame()
df_valid['Sequence'] = list(top_prots)

def create_3d_point(sequence, c):
    # Step 1: Count the number of occurrences of character c
    normalized_chars = ['X','U','B','Z','O','J']

    for norm_char in normalized_chars:
        sequence = sequence.replace(norm_char, '')

    count_c = sequence.count(c)
    
    if count_c == 0:
        return [0, 0, 0]  # Return None if the character c is not found
    
    # Step 2: Calculate the mean distance from the start of all occurrences of c
    indices_c = [i for i, char in enumerate(sequence) if char == c]
    mean_distance = sum(indices_c) / count_c
    
    # Step 3: Compute the second normalized central moment of the distribution of c
    variance = sum((idx - mean_distance) ** 2 for idx in indices_c) / count_c
    second_moment = variance / (len(sequence))
    
    # Create the 3D vector
    point = [count_c, mean_distance, second_moment]
    
    return point

def point_in_hull(point, hull_equations, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull_equations)

# Iterate over all arrays inside the file
def check_validity(seq):
    inside_hulls = []
    for c in data.files:
        equations = data[c]
        point = create_3d_point(seq, c)
        inside_hulls.append(point_in_hull(point, equations))
    return all(inside_hulls)


print('Starting Protein Validity Check...')
df_valid['Valid Protein'] = [check_validity(seq) for seq in top_prots]

filtered_prots = [prot for prot in top_prots if check_validity(prot)]

print('Valid proteins identified.')
del model
del protgpt2
gc.collect()
torch.cuda.empty_cache()
print('Starting structure check...')
esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")



# Define quantization configuration
# quantization_config = QuantoConfig(weights="int8")
# esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", quantization_config=quantization_config)

# Quantize the model


esm_model = esm_model.to(device)

plddt_results = []


for sequence in tqdm(top_prots):
    # Input the sequence to ESM model
    input_sequence = sequence.replace('\n', '')
    if len(input_sequence) == 0:
        continue
    inputs = esm_tokenizer([input_sequence], return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to(device)
    outputs = esm_model(**inputs)
    avg_plddt = torch.mean(outputs.plddt)
    plddt_results.append(avg_plddt.item())
    torch.cuda.empty_cache()


df_valid['plDDT'] = plddt_results
plddt_results = np.array(plddt_results)
df_valid['Good Structure'] = list(plddt_results > 0.7)

#Save valid protein and structure check data to csv
df_valid.to_csv(args.output_dir + '/generation_checks.csv', index=False)

plddt_results = np.array([plddt_results[i] for i in range(len(top_prots)) if check_validity(top_prots[i])])

filtered_prots = np.array(filtered_prots)

best_prots = filtered_prots[plddt_results > 0.7]

print('Structure check complete. ')



del esm_model
gc.collect()
torch.cuda.empty_cache()


#Loading PeptideBERT model

from PeptideBERT.data.dataloader import load_data
from PeptideBERT.model.network import create_model, cri_opt_sch
from PeptideBERT.model.utils import train, validate, test


save_dir = args.pred_model_path

config = yaml.load(open(save_dir + '/config.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device

peptideBERT_model = create_model(config)


peptideBERT_model.load_state_dict(torch.load(f'{save_dir}/model.pt')['model_state_dict'], strict=False)


m2 = dict(zip(
    ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','L',
    'A','G','V','E','S','I','K','R','D','T','P','N',
    'Q','F','Y','M','H','C','W','X','U','B','Z','O'],
    range(30)
))

def f(seq):
    seq = map(lambda x: m2[x], seq)
    seq = torch.tensor([*seq], dtype=torch.long).unsqueeze(0).to(device)
    attn = torch.tensor(seq > 0, dtype=torch.long).to(device)

    return seq, attn

df_preds = pd.DataFrame()

scores = {}

print('Starting protein property predictions...')

peptides_to_exclude = "XUBZOJ"

best_prots = remove_prots_with(best_prots, peptides_to_exclude)


for seq in tqdm(best_prots):

    

    seq = seq.replace('\n', '')



    score = peptideBERT_model(*f(seq)).item()

    scores[seq] = score

print('Property prediction completed.')

df_preds['Sequence'] = scores.keys()
df_preds['Score'] = scores.values()
df_preds['Property'] = df_preds['Score'] > 0.5

total_proteins_with_property = np.sum(np.array(list(scores.values())) > 0.5)

property_probability = total_proteins_with_property/len(best_prots)

# Save Predictions to CSV
df_preds.to_csv(args.output_dir + '/predictions.csv', index=False)

print('Inference Complete')

with open(args.output_dir + '/info.txt', 'w') as file:
    # Write 'print('hello world')' to the file
    file.write(f'Total generated sequences {args.num_return_sequences} out of which top {args.num_return_sequences//3} sequences based on perplexity were chosen.\n')
    file.write(f'Total sequences which passed protein validty check {len(filtered_prots)}, rejected {len(top_prots) - len(filtered_prots)} or {((len(top_prots) - len(filtered_prots))/len(top_prots) * 100):.3f}%\n')
    file.write(f'Total proteins which had plDDT > 0.7 from ESMFold are {len(best_prots)}, {(len(best_prots)/len(filtered_prots)*100):.3f}% of valid generated proteins\n')
    file.write(f'{total_proteins_with_property}/{len(best_prots)} had the desired property, which is {property_probability:.4f} probability.\n')
    for arg in vars(args):
        file.write(f"{arg}: {getattr(args, arg)}\n")

print(f'Total generated sequences {args.num_return_sequences} out of which top {args.num_return_sequences//3} sequences based on perplexity were chosen.')
print(f'Total sequences which passed protein validty check {len(filtered_prots)}, rejected {len(top_prots) - len(filtered_prots)} or {((len(top_prots) - len(filtered_prots))/len(top_prots) * 100):.3f}%')
print(f'Total proteins which had plDDT > 0.7 from ESMFold are {len(best_prots)}, {(len(best_prots)/len(filtered_prots) * 100):.3f}% of valid generated proteins')
print(f'{total_proteins_with_property}/{len(best_prots)} had the desired property, which is {property_probability:.4f} probability.')
