from datasets import load_dataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import spacy


class Tokenizer:
    def __init__(self):
        self.spacy_de = spacy.load('de_core_news_sm')  # German
        self.spacy_en = spacy.load('en_core_web_sm')  # English

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

class HuggingFaceDatasetWrapper(torch.utils.data.Dataset):
    """
    Wraps a Hugging Face Dataset object to make it compatible with PyTorch DataLoader.
    """
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class Multi30kEnDeLoader:
    def __init__(self, tokenize_en, tokenize_de, init_token, eos_token, pad_token="<pad>"):
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.source_vocab = {}
        self.target_vocab = {}
        self.reverse_source_vocab = {}
        self.reverse_target_vocab = {}

    def decode(self, token_ids, reverse_vocab):
        """
        Converts a list of token IDs back into a string using the reverse vocabulary.
        """
        tokens = [reverse_vocab[token_id] for token_id in token_ids if token_id in reverse_vocab]
        return " ".join(tokens)

    def build_vocab(self, data, tokenize, min_freq=2):
        token_counts = {}
        for sentence in data:
            for token in tokenize(sentence.lower()):
                token_counts[token] = token_counts.get(token, 0) + 1

        vocab = {self.pad_token: 0, self.init_token: 1, self.eos_token: 2}
        for token, count in token_counts.items():
            if count >= min_freq:
                vocab[token] = len(vocab)

        return vocab

    def preprocess_data(self, data, tokenize, vocab, max_len=128, add_special_tokens=False, subsample=False):
        """
        Tokenizes and converts sentences to indices.
        Optionally applies subsampling to reduce frequent words.
        For target sequences, adds <bos> and <eos> if `add_special_tokens` is True.
        """
        sequences = []  # Initialize the list to store processed sequences
        # common_tokens = {" ", ",", ".", ":", ";", "-", "_", "(", ")", "[", "]", "{", "}", "!"}  # Add more as needed
        common_tokens = {"."}
        for sentence in data:
            # Tokenize the input
            tokens = tokenize(sentence.lower())

            # Remove common tokens
            # tokens = [token for token in tokens if token not in common_tokens]

            # Truncate tokens if necessary
            tokens = tokens[:max_len - (2 if add_special_tokens else 0)]

            # Add special tokens for target sequences
            if add_special_tokens:
                tokens = [self.init_token] + tokens + [self.eos_token]

            # Convert tokens to indices
            indices = [vocab.get(token, vocab[self.pad_token]) for token in tokens]
            sequences.append(indices)

        return sequences

    def collate_fn(self, batch):
        src = [torch.tensor(item["src"]) for item in batch]
        trg = [torch.tensor(item["trg"]) for item in batch]

        src_padded = pad_sequence(src, batch_first=True, padding_value=self.source_vocab[self.pad_token])
        trg_padded = pad_sequence(trg, batch_first=True, padding_value=self.target_vocab[self.pad_token])

        return {"input_ids": src_padded, "labels": trg_padded}

    def create_dataloader(self, split, batch_size=32, shuffle=False, max_len=128,num_workers=0):
        dataset = load_dataset("bentrevett/multi30k", split=split)

        if not self.source_vocab:
            self.source_vocab = self.build_vocab(dataset["en"], self.tokenize_en)
            self.target_vocab = self.build_vocab(dataset["de"], self.tokenize_de)
            self.reverse_source_vocab = {v: k for k, v in self.source_vocab.items()}
            self.reverse_target_vocab = {v: k for k, v in self.target_vocab.items()}

        dataset = dataset.map(
            lambda examples: {
                # Encoder input without <bos> or <eos>
                "src": self.preprocess_data([examples["en"]], self.tokenize_en, self.source_vocab, max_len,
                                            subsample=False)[0],
                # Decoder input and output with <bos> and <eos>
                "trg": self.preprocess_data([examples["de"]], self.tokenize_de, self.target_vocab, max_len,
                                            add_special_tokens=True, subsample=False)[0],
            },
            batched=False,
        )

        # Wrap the Hugging Face dataset in a PyTorch Dataset wrapper
        wrapped_dataset = HuggingFaceDatasetWrapper(dataset)

        return TorchDataLoader(wrapped_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, num_workers=num_workers)
