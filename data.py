from transformers import AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Load Tokenizer
src_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")  # English tokenizer
trg_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-french-europeana-cased")  # French tokenizer

# Load CSV
def load_csv_dataset(file_path):
    data = pd.read_csv(file_path)
    return data["English"].tolist(), data["French"].tolist()

# Preprocess Dataset
def preprocess(sentences, tokenizer, max_len=128):
    """
    Tokenizes and converts sentences to input IDs using a prebuilt tokenizer.
    Includes padding and truncation for uniform lengths.
    """
    tokenized = tokenizer(
        sentences,
        max_length=max_len,
        padding="max_length",  # Ensures all sentences are padded to max_len
        truncation=True,       # Truncates sentences longer than max_len
        return_tensors="pt",   # Returns PyTorch tensors
    )
    return tokenized["input_ids"]


# Custom Dataset
class TranslationDataset(Dataset):
    def __init__(self, src_data, trg_data):
        self.src_data = src_data
        self.trg_data = trg_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return {"src": self.src_data[idx], "trg": self.trg_data[idx]}

# Collate Function
def collate_fn(batch):
    src = [item["src"] for item in batch]
    trg = [item["trg"] for item in batch]

    src_padded = pad_sequence(src, batch_first=True, padding_value=src_tokenizer.pad_token_id)
    trg_padded = pad_sequence(trg, batch_first=True, padding_value=trg_tokenizer.pad_token_id)

    return {"input_ids": src_padded, "labels": trg_padded}

# Load and Preprocess Dataset
file_path = "english_french.csv"  # Replace with your dataset path
english_sentences, french_sentences = load_csv_dataset(file_path)

src_data = preprocess(english_sentences, src_tokenizer)
trg_data = preprocess(french_sentences, trg_tokenizer)

# Split into train/validation/test
train_size = int(0.8 * len(src_data))
valid_size = int(0.1 * len(src_data))

train_src, train_trg = src_data[:train_size], trg_data[:train_size]
valid_src, valid_trg = src_data[train_size:train_size+valid_size], trg_data[train_size:train_size+valid_size]
test_src, test_trg = src_data[train_size+valid_size:], trg_data[train_size+valid_size:]

# Create DataLoaders
train_dataset = TranslationDataset(train_src, train_trg)
valid_dataset = TranslationDataset(valid_src, valid_trg)
test_dataset = TranslationDataset(test_src, test_trg)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Example Usage
for batch in train_loader:
    print(batch)
    break

# Vocabulary Info
print(f"Source Tokenizer Vocabulary Size: {src_tokenizer.vocab_size}")
print(f"Target Tokenizer Vocabulary Size: {trg_tokenizer.vocab_size}")
