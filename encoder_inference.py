import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import pandas as pd
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

# torch.cuda.empty_cache()

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

parser = argparse.ArgumentParser(description='')
    
# Adding arguments
parser.add_argument('--model_name', type=str)
parser.add_argument('--experiment', type=str)
args = parser.parse_args()

model_name = args.model_name
experiment = args.experiment



# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, max_length=512):  # You can set max_length to an appropriate value

        self.max_length = max_length
        self.texts = texts
        self.labels = labels

        

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(
            self.texts[idx],
            return_tensors='pt',
            # truncation=True,
            padding='max_length',  # Use padding to ensure all sequences have the same length
            max_length=self.max_length
        )

        is_truncated = False
        if len(inputs['input_ids'][0]) > self.max_length:
            is_truncated = True

        inputs = tokenizer(
            self.texts[idx],
            return_tensors='pt',
            truncation=True,
            padding='max_length',  # Use padding to ensure all sequences have the same length
            max_length=self.max_length
        )
           
        label = torch.tensor(self.labels[idx])
        return inputs, label, is_truncated

def collate_fn(batch):
    inputs, labels, is_truncated = zip(*batch)
    return {
        'input_ids': torch.stack([x['input_ids'].squeeze() for x in inputs]),
        'attention_mask': torch.stack([x['attention_mask'].squeeze() for x in inputs]),
        'labels': torch.tensor(labels), 
        'is_truncated': is_truncated
    }


if experiment == "evasion_based_clarity":
    num_labels = 9
    mapping_labels = {'Explicit': 0, 'Implicit': 1, 'Dodging': 2, 'Deflection': 3, 'Partial': 4, 'General': 5, 'Declining': 6, 'Ignorance': 7, 'Clarification': 8}
    label = "evasion_label"
elif experiment == "direct_clarity":
    num_labels = 3
    mapping_labels = {'Clear Reply': 0, "Ambivalent": 1, "Clear Non-Reply": 2}
    label = "clarity_label"


# --- Load Model and Tokenizer ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print(f"Loading model and tokenizer for: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=num_labels
).to(device)

# --- Set max_size based on model type (if needed) ---
# We still need to handle XLNet's different input size, but that's it.
if "xlnet" in model_name:
    max_size = 4096
else:
    max_size = 512

print(f"Model {model_name} loaded. Max sequence length set to {max_size}.")


dataset = load_dataset("ailsntua/QEvasion")
all_texts = [f"Question: {row['interview_question']}\n\nAnswer: {row['interview_answer']}\n\nSubanswer: {row['question']}" for row in dataset['test']]
all_labels = [mapping_labels[row[label]] for row in dataset['test']]

# Create datasets and dataloaders
val_dataset = CustomDataset(all_texts, all_labels, max_length=512)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Inside the validation loop
import numpy as np

model.eval()
inv_mapping_labels = {v:k for k, v in mapping_labels.items()}
results = []

true_labels, pred_labels = [], []
with torch.no_grad():
    for batch in tqdm(val_dataloader):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        is_truncated = batch['is_truncated']
        
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        for true_label, pred_label, is_trunc in zip(labels.cpu().numpy(), outputs["logits"].cpu().numpy(), is_truncated):
            true_label = inv_mapping_labels[true_label]
            pred_label = inv_mapping_labels[np.argmax(pred_label)]
            results.append([is_trunc, true_label, pred_label])

df = pd.DataFrame(results, columns=['is_truncated', 'true_labels', 'pred_labels'])
os.makedirs("./results", exist_ok=True)
df.to_csv(f"./results/{model_name.split('/')[-1]}-{experiment}.csv")

