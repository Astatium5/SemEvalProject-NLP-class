import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import argparse
from collections import Counter


# import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

import torch
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


# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.max_length = max_length
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for text, label in zip(texts, labels):
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            # enc["input_ids"]: (1, max_length) → squeeze(0) = (max_length)
            self.input_ids.append(enc["input_ids"].squeeze(0))
            self.attention_masks.append(enc["attention_mask"].squeeze(0))
            self.labels.append(torch.tensor(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


# Example data

dataset = load_dataset("ailsntua/QEvasion")
all_texts = [f"Question: {row['interview_question']}\n\nAnswer: {row['interview_answer']}\n\nSubanswer: {row['question']}" for row in dataset['train']]
all_labels = [mapping_labels[row[label]] for row in dataset['train']]
print (set(all_labels))
print (len(all_texts))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_texts,
    all_labels,
    test_size=0.1,      # 10% validation
    random_state=42,    # reproducibility
    stratify=all_labels # stratify by labels
)

# Create datasets and dataloaders
train_dataset = CustomDataset(train_texts, train_labels, max_length=512, tokenizer=tokenizer)
val_dataset = CustomDataset(val_texts, val_labels, max_length=512, tokenizer=tokenizer)

label_counts = Counter(all_labels)
print(label_counts)
num_labels = len(mapping_labels)
total = sum(label_counts.values())
class_weights = torch.tensor(
    [total / label_counts[i] for i in range(num_labels)], 
    dtype=torch.float
).to(device)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

print("Class weights:", class_weights)

sample_weights = [1.0 / label_counts[label] for label in train_labels]
sample_weights = torch.tensor(sample_weights, dtype=torch.float)

# balancing batch sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
val_dataloader = DataLoader(val_dataset, batch_size=4)


print (len(train_dataloader), len(val_dataloader))

# Fine-tuning
optimizer = AdamW(model.parameters(), lr=1e-5)

num_epochs = 10
for epoch in range(num_epochs):
    # Training
    model.train()
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/ {num_epochs} - Training'):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
    
        outputs = model(input_ids=inputs, attention_mask=attention_mask)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Inside the validation loop
    model.eval()
    val_loss = 0.0
    
    pred_labels = [] 
    true_labels = [] 
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    
            # Calculate accuracy
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            pred_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            pred_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())


    average_val_loss = val_loss / len(val_dataloader)
    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')

    print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {average_val_loss:.4f} - Accuracy: {accuracy * 100:.2f}% - Macro F1 Score: {macro_f1:.4f}')

# Save the fine-tuned model

out_file = f"{model_name.split('/')[-1]}-qaevasion-{experiment}"
model.save_pretrained(out_file)
