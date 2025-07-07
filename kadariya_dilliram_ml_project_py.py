import pandas as pd

# Load train.csv
train_df = pd.read_csv("/content/drive/MyDrive/ML project NLP/train_data.csv")
display(train_df)

# Load test.csv
test_df = pd.read_csv("/content/drive/MyDrive/ML project NLP/test_data.csv")
display(test_df)

# train and test on a subset of data
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
train_data = train_df.iloc[:20000]

pip install transformers datasets

from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import os
os.environ["WANDB_DISABLED"] = "true"


# Split the train.csv into into train (85%), validation (15%)
train_data, val_data = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df["label"])


# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    texts = [h + " " + d for h, d in zip(examples["headline"], examples["description"])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize and remove text columns
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["headline", "description"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["headline", "description"])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["headline", "description"])

# Set the format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4) # uncased vs cased

# model hyperparameters and arguments
training_args = TrainingArguments(
    output_dir="/content/results",                  # Where to save model checkpoints
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="/content/logs",                   # For TensorBoard logs
    save_strategy="epoch",                         # Save after each epoch
    save_total_limit=1,                            # Keep only the last checkpoint
    report_to="none"                               # Disable W&B or other integrations
)

# Trainer without metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# ========================
# Now extract ANN outputs
# ========================

def extract_features(model, dataset):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=32)
    features, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            outputs = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token output
            features.append(cls_output.cpu().numpy())
            labels.append(batch["label"].numpy())

    return np.vstack(features), np.concatenate(labels)

# Extract [CLS] features
train_features, train_labels = extract_features(model, train_dataset)
test_features, test_labels = extract_features(model, test_dataset)

# Train and evaluate KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_features, train_labels)
knn_preds = knn.predict(test_features)
knn_acc = accuracy_score(test_labels, knn_preds)

print("KNN Accuracy:", knn_acc)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from datasets import Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# Load pretrained tokenizer and trained model
# =====================================================
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "/content/drive/MyDrive/ML final project NLP/distilbert_model_uncased"
)
model.eval()  # Set model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =====================================================
# Tokenization
# =====================================================
def tokenize_function(examples):
    texts = [h + " " + d for h, d in zip(examples["headline"], examples["description"])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_df)

# Remove original text columns after tokenizing
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["headline", "description"])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["headline", "description"])

# Set PyTorch format
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# =====================================================
# Extract [CLS] token embeddings
# =====================================================
def extract_features(model, dataset):
    dataloader = DataLoader(dataset, batch_size=32)
    features, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            features.append(cls_output.cpu().numpy())
            labels.append(batch["label"].cpu().numpy())

    return np.vstack(features), np.concatenate(labels)

# Extract ANN outputs
train_features, train_labels = extract_features(model, train_dataset)
test_features, test_labels = extract_features(model, test_dataset)

# =====================================================
# Train and evaluate KNN
# =====================================================
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(train_features, train_labels)
knn_preds = knn.predict(test_features)
knn_acc = accuracy_score(test_labels, knn_preds)

print(" KNN Accuracy:", knn_acc)
