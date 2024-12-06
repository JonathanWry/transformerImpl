import os
import requests
import gzip
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy
import torch
from collections import Counter



# Step 1: Download and Extract Data
def download_and_unpack(url, output_path):
    response = requests.get(url)
    with open(output_path, "wb") as f:
        f.write(response.content)
    if output_path.endswith(".gz"):
        with gzip.open(output_path, "rt", encoding="utf-8") as gz_file:
            with open(output_path.rstrip(".gz"), "w", encoding="utf-8") as output_file:
                output_file.write(gz_file.read())
        os.remove(output_path)


def download_multi30k(data_dir="multi30k_data"):
    os.makedirs(data_dir, exist_ok=True)
    urls = {
        "train_en": "https://github.com/multi30k/dataset/raw/master/data/task1/raw/train.en.gz",
        "train_fr": "https://github.com/multi30k/dataset/raw/master/data/task1/raw/train.fr.gz",
        "val_en": "https://github.com/multi30k/dataset/raw/master/data/task1/raw/val.en.gz",
        "val_fr": "https://github.com/multi30k/dataset/raw/master/data/task1/raw/val.fr.gz",
        "test_en": "https://github.com/multi30k/dataset/raw/master/data/task1/raw/test_2016_flickr.en.gz",
        "test_fr": "https://github.com/multi30k/dataset/raw/master/data/task1/raw/test_2016_flickr.fr.gz",
    }
    for key, url in urls.items():
        output_path = os.path.join(data_dir, os.path.basename(url))
        if not os.path.exists(output_path.rstrip(".gz")):
            download_and_unpack(url, output_path)
            print(f"Downloaded and unpacked: {output_path}")
        else:
            print(f"File already exists: {output_path}")


# Custom Dataset Class
class TranslationDataset(Dataset):
    def __init__(self, src_data, trg_data, src_vocab, trg_vocab, max_len):
        self.src_data = src_data
        self.trg_data = trg_data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_sentence = self.process_sentence(self.src_data[idx], self.src_vocab)
        trg_sentence = self.process_sentence(self.trg_data[idx], self.trg_vocab, add_special_tokens=True)
        return {"src": torch.tensor(src_sentence), "trg": torch.tensor(trg_sentence)}

    def process_sentence(self, sentence, vocab, add_special_tokens=False):
        tokens = [vocab.get(token, vocab["<unk>"]) for token in sentence.split()]
        if add_special_tokens:
            tokens = [vocab["<bos>"]] + tokens[:self.max_len - 2] + [vocab["<eos>"]]
        else:
            tokens = tokens[:self.max_len]
        return tokens


# DataLoader with Additional Utilities
class Multi30kEnFrLoader:
    def __init__(self, tokenize_src, tokenize_trg, init_token="<bos>", eos_token="<eos>", pad_token="<pad>"):
        self.tokenize_src = tokenize_src
        self.tokenize_trg = tokenize_trg
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.source_vocab = {}
        self.target_vocab = {}
        self.reverse_source_vocab = {}
        self.reverse_target_vocab = {}

    def build_vocab(self, data, tokenize, min_freq=2, filter_tokens=None):
        word_freq = Counter()
        for sentence in data:
            tokens = tokenize(sentence.lower())
            if filter_tokens:
                tokens = [token for token in tokens if token not in filter_tokens]
            word_freq.update(tokens)

        vocab = {self.pad_token: 0, self.init_token: 1, self.eos_token: 2, "<unk>": 3}
        for word, freq in word_freq.items():
            if freq >= min_freq:
                vocab[word] = len(vocab)

        reverse_vocab = {idx: word for word, idx in vocab.items()}
        return vocab, reverse_vocab

    def preprocess_data(self, data, tokenize, vocab, max_len, add_special_tokens=False):
        sequences = []
        for sentence in data:
            tokens = tokenize(sentence.lower())
            tokens = tokens[:max_len - (2 if add_special_tokens else 0)]
            if add_special_tokens:
                tokens = [self.init_token] + tokens + [self.eos_token]
            indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
            sequences.append(indices)
        return sequences

    def decode(self, token_ids, reverse_vocab):
        """
        Converts a list of token IDs back into a string using the reverse vocabulary.
        """
        tokens = [reverse_vocab.get(tok, "<unk>") for tok in token_ids]
        return " ".join(tokens)

    def create_dataloader(self, src_data, trg_data, max_len, batch_size, shuffle=True,num_workers=0):
        dataset = TranslationDataset(src_data, trg_data, self.source_vocab, self.target_vocab, max_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn,num_workers=num_workers)

    def collate_fn(self, batch):
        src = [torch.tensor(item["src"]) for item in batch]
        trg = [torch.tensor(item["trg"]) for item in batch]
        src_padded = pad_sequence(src, batch_first=True, padding_value=self.source_vocab[self.pad_token])
        trg_padded = pad_sequence(trg, batch_first=True, padding_value=self.target_vocab[self.pad_token])
        return {"input_ids": src_padded, "labels": trg_padded}


# Main Script
if __name__ == "__main__":
    # Download dataset
    data_dir = "multi30k_data"
    download_multi30k(data_dir)

    # Load datasets
    with open(os.path.join(data_dir, "train.en"), "r", encoding="utf-8") as f:
        train_en = f.readlines()
    with open(os.path.join(data_dir, "train.fr"), "r", encoding="utf-8") as f:
        train_fr = f.readlines()
    with open(os.path.join(data_dir, "val.en"), "r", encoding="utf-8") as f:
        val_en = f.readlines()
    with open(os.path.join(data_dir, "val.fr"), "r", encoding="utf-8") as f:
        val_fr = f.readlines()
    with open(os.path.join(data_dir, "test_2016_flickr.en"), "r", encoding="utf-8") as f:
        test_en = f.readlines()
    with open(os.path.join(data_dir, "test_2016_flickr.fr"), "r", encoding="utf-8") as f:
        test_fr = f.readlines()

    # Initialize tokenizers
    spacy_en = spacy.load("en_core_web_sm")
    spacy_fr = spacy.load("fr_core_news_sm")
    tokenize_en = lambda text: [tok.text.lower() for tok in spacy_en.tokenizer(text)]
    tokenize_fr = lambda text: [tok.text.lower() for tok in spacy_fr.tokenizer(text)]

    # Initialize CustomDataLoader
    loader = Multi30kEnFrLoader(tokenize_en, tokenize_fr)

    # Build vocabularies with filtering
    common_tokens = {}  # Add common tokens to filter out
    # common_tokens = {}
    loader.source_vocab, loader.reverse_source_vocab = loader.build_vocab(train_en, tokenize_en, filter_tokens=common_tokens)
    loader.target_vocab, loader.reverse_target_vocab = loader.build_vocab(train_fr, tokenize_fr, filter_tokens=common_tokens)

    # Create DataLoaders
    train_loader = loader.create_dataloader(train_en, train_fr, max_len, batch_size, shuffle=True)
    valid_loader = loader.create_dataloader(val_en, val_fr, max_len, batch_size, shuffle=False)
    test_loader = loader.create_dataloader(test_en, test_fr, max_len, batch_size, shuffle=False)

    # Debugging
    print("Source Vocabulary Size:", len(loader.source_vocab))
    print("Target Vocabulary Size:", len(loader.target_vocab))

    # Print a decoded example batch
    for batch in train_loader:
        print("Decoded Source:", loader.decode(batch["input_ids"][0].tolist(), loader.reverse_source_vocab))
        print("Decoded Target:", loader.decode(batch["labels"][0].tolist(), loader.reverse_target_vocab))
        break
    enc_vocab_size = len(loader.source_vocab)  # Vocabulary size for the encoder (source)
    dec_vocab_size = len(loader.target_vocab)
    pad_token=loader.source_vocab["<pad>"]
