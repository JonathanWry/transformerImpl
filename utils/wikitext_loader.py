import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from collections import Counter

# Custom DataLoader with Utilities
class WikitextLoader:
    def __init__(self, tokenize_src, tokenize_trg, common_tokens=None, init_token="<bos>", eos_token="<eos>",
                 pad_token="<pad>"):
        self.tokenize_src = tokenize_src
        self.tokenize_trg = tokenize_trg
        self.common_tokens = common_tokens if common_tokens else {}
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.source_vocab = {}
        self.target_vocab = {}
        self.reverse_source_vocab = {}
        self.reverse_target_vocab = {}

    def build_vocab(self, data_iter, tokenize, min_freq=2):
        """
        Builds vocabulary from the dataset iterator, excluding common tokens.
        """
        word_freq = Counter()
        for sentence in data_iter:
            tokens = tokenize(sentence["text"].lower())
            # Filter out common tokens
            tokens = [token for token in tokens if token not in self.common_tokens]
            word_freq.update(tokens)

        vocab = {self.pad_token: 0, self.init_token: 1, self.eos_token: 2, "<unk>": 3}
        for word, freq in word_freq.items():
            if freq >= min_freq:
                vocab[word] = len(vocab)

        reverse_vocab = {idx: token for token, idx in vocab.items()}
        return vocab, reverse_vocab

    def preprocess_data(self, data_iter, tokenize, vocab, max_len=256):
        """
        Processes the input data iterator:
        - Splits sentences into source and target halves.
        - Applies truncation logic: source uses last max_len tokens, target uses first max_len tokens.
        """
        source_sequences, target_sequences = [], []
        for sentence in data_iter:
            if isinstance(sentence, dict) and "text" in sentence:  # Ensure correct access for raw dataset
                text = sentence["text"]
            elif isinstance(sentence, str):  # Handle raw string data
                text = sentence
            else:
                raise ValueError("Unexpected sentence format in preprocess_data.")

            # Tokenize the full sentence
            tokens = tokenize(text.lower())
            # Filter out common tokens
            tokens = [token for token in tokens if token not in self.common_tokens]

            # Split tokens into source (first half) and target (second half)
            midpoint = len(tokens) // 2
            source_tokens = tokens[:midpoint][-max_len + 2:]  # Last max_len tokens of the first half
            target_tokens = tokens[midpoint:][:max_len - 2]  # First max_len tokens of the second half

            # Convert to indices and add special tokens
            source_indices = [vocab.get("<bos>", 1)] + [vocab.get(token, vocab["<unk>"]) for token in source_tokens] + [
                vocab.get("<eos>", 2)]
            target_indices = [vocab.get("<bos>", 1)] + [vocab.get(token, vocab["<unk>"]) for token in target_tokens] + [
                vocab.get("<eos>", 2)]

            source_sequences.append(source_indices)
            target_sequences.append(target_indices)

        return source_sequences, target_sequences

    def decode(self, token_ids, reverse_vocab):
        """
        Converts a list of token IDs back to a string using reverse vocabulary.
        """
        return " ".join(reverse_vocab.get(tok, "<unk>") for tok in token_ids)

    def collate_fn(self, batch):
        """
        Pads source and target sequences dynamically to match the longest sequence in the batch.
        """
        src = [torch.tensor(item[0]) for item in batch]
        trg = [torch.tensor(item[1]) for item in batch]

        if len(src) != len(trg):
            raise ValueError("Mismatch in the number of source and target sequences in the batch.")

        pad_token = self.source_vocab[self.pad_token]
        src_padded = pad_sequence(src, batch_first=True, padding_value=pad_token)
        trg_padded = pad_sequence(trg, batch_first=True, padding_value=pad_token)

        return {"input_ids": src_padded, "labels": trg_padded}

    def create_dataloader(self, src_data, trg_data, batch_size, shuffle=False):
        """
        Creates a DataLoader with the given source and target datasets.
        """
        if len(src_data) != len(trg_data):
            raise ValueError("Source and Target datasets must have the same number of samples.")

        dataset = list(zip(src_data, trg_data))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )


if __name__ == "__main__":
    # Load datasets from WikiText-2 (used here as an example)
    train_iter = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    valid_iter = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    test_iter = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Filter empty lines
    train_iter = train_iter.filter(lambda x: x["text"].strip() != "")
    valid_iter = valid_iter.filter(lambda x: x["text"].strip() != "")
    test_iter = test_iter.filter(lambda x: x["text"].strip() != "")

    # Tokenizers
    tokenizer_src = get_tokenizer("basic_english")
    tokenizer_trg = get_tokenizer("basic_english")

    # Initialize DataLoader
    # common_tokens = {".", ",", "=", "!", "-", "_", "(", ")", "[", "]", "{", "}"}
    common_tokens={}

    # Initialize CustomDataLoader
    loader = WikitextLoader(tokenizer_src, tokenizer_trg, common_tokens)

    # Build vocabularies with filtering
    loader.source_vocab, loader.reverse_source_vocab = loader.build_vocab(train_iter, tokenizer_src)
    loader.target_vocab, loader.reverse_target_vocab = loader.build_vocab(train_iter, tokenizer_trg)

    # Preprocess data
    train_data_src, train_data_trg = loader.preprocess_data(train_iter, tokenizer_src, loader.source_vocab, max_len=max_len)
    valid_data_src, valid_data_trg = loader.preprocess_data(valid_iter, tokenizer_src, loader.source_vocab, max_len=max_len)
    test_data_src, test_data_trg = loader.preprocess_data(test_iter, tokenizer_src, loader.source_vocab, max_len=max_len)

    # Create DataLoaders
    train_loader = loader.create_dataloader(train_data_src, train_data_trg, batch_size, shuffle=True)
    valid_loader = loader.create_dataloader(valid_data_src, valid_data_trg, batch_size)
    test_loader = loader.create_dataloader(test_data_src, test_data_trg, batch_size)

    # Debugging
    print("Source Vocabulary Size:", len(loader.source_vocab))
    print("Target Vocabulary Size:", len(loader.target_vocab))

    # Print a decoded example batch
    for batch in train_loader:
        print("Input IDs Shape:", batch["input_ids"].shape)
        print("Labels Shape:", batch["labels"].shape)
        break
    for batch in train_loader:
        # Decode the first source sequence in the batch
        decoded_source = loader.decode(batch["input_ids"][0].tolist(), loader.reverse_source_vocab)
        # Decode the first target sequence in the batch
        decoded_target = loader.decode(batch["labels"][0].tolist(), loader.reverse_target_vocab)

        print("Decoded Source:", decoded_source)
        print("Decoded Target:", decoded_target)
        break

    enc_vocab_size = len(loader.source_vocab)
    dec_vocab_size = len(loader.target_vocab)
    pad_token = loader.source_vocab["<pad>"]
