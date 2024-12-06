import spacy
from torchtext.data.utils import get_tokenizer
from datasets import load_dataset

from utils.multi30k_en_fr_loader import Multi30kEnFrLoader, download_multi30k
from utils.multi30k_en_de_loader import Multi30kEnDeLoader
from utils.wikitext_loader import WikitextLoader


def get_loader_and_dataloaders(dataset_name, batch_size=32, max_len=128):
    # Initialize tokenizers
    spacy_en = spacy.load("en_core_web_sm")
    spacy_de = spacy.load("de_core_news_sm")
    spacy_fr = spacy.load("fr_core_news_sm")
    tokenizer_src = get_tokenizer("basic_english")
    tokenizer_trg = get_tokenizer("basic_english")

    if dataset_name == "multi30k_en_de":
        # Multi30k English-German loader
        loader = Multi30kEnDeLoader(
            tokenize_en=lambda text: [tok.text.lower() for tok in spacy_en.tokenizer(text)],
            tokenize_de=lambda text: [tok.text.lower() for tok in spacy_de.tokenizer(text)],
            init_token="<bos>",
            eos_token="<eos>",
        )
        # Create DataLoaders
        train_loader = loader.create_dataloader(split="train", batch_size=batch_size, max_len=max_len,num_workers=0)
        valid_loader = loader.create_dataloader(split="validation", batch_size=batch_size, max_len=max_len,num_workers=0)
        test_loader = loader.create_dataloader(split="test", batch_size=batch_size, max_len=max_len,num_workers=0)
        return loader, train_loader, valid_loader, test_loader

    elif dataset_name == "multi30k_en_fr":
        # Multi30k English-French loader
        data_dir = "multi30k_data"
        download_multi30k(data_dir)

        # Load datasets
        with open(f"{data_dir}/train.en", "r", encoding="utf-8") as f:
            train_en = f.readlines()
        with open(f"{data_dir}/train.fr", "r", encoding="utf-8") as f:
            train_fr = f.readlines()
        with open(f"{data_dir}/val.en", "r", encoding="utf-8") as f:
            val_en = f.readlines()
        with open(f"{data_dir}/val.fr", "r", encoding="utf-8") as f:
            val_fr = f.readlines()
        with open(f"{data_dir}/test_2016_flickr.en", "r", encoding="utf-8") as f:
            test_en = f.readlines()
        with open(f"{data_dir}/test_2016_flickr.fr", "r", encoding="utf-8") as f:
            test_fr = f.readlines()

        # Initialize loader
        loader = Multi30kEnFrLoader(
            tokenize_src=lambda text: [tok.text.lower() for tok in spacy_en.tokenizer(text)],
            tokenize_trg=lambda text: [tok.text.lower() for tok in spacy_fr.tokenizer(text)],
            init_token="<bos>",
            eos_token="<eos>",
        )

        # Build vocabularies
        loader.source_vocab, loader.reverse_source_vocab = loader.build_vocab(train_en, loader.tokenize_src)
        loader.target_vocab, loader.reverse_target_vocab = loader.build_vocab(train_fr, loader.tokenize_trg)

        # Create DataLoaders
        train_loader = loader.create_dataloader(train_en, train_fr, max_len, batch_size,num_workers=0)
        valid_loader = loader.create_dataloader(val_en, val_fr, max_len, batch_size,num_workers=0)
        test_loader = loader.create_dataloader(test_en, test_fr, max_len, batch_size,num_workers=0)
        return loader, train_loader, valid_loader, test_loader

    elif dataset_name == "wikitext":
        # Wikitext loader
        loader = WikitextLoader(
            tokenize_src=tokenizer_src,
            tokenize_trg=tokenizer_trg,
        )
        # Load datasets
        train_iter = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",num_workers=0)
        valid_iter = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",num_workers=0)
        test_iter = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",num_workers=0)

        # Filter empty lines
        train_iter = train_iter.filter(lambda x: x["text"].strip() != "")
        valid_iter = valid_iter.filter(lambda x: x["text"].strip() != "")
        test_iter = test_iter.filter(lambda x: x["text"].strip() != "")

        # Build vocabularies
        loader.source_vocab, loader.reverse_source_vocab = loader.build_vocab(train_iter, tokenizer_src)
        loader.target_vocab, loader.reverse_target_vocab = loader.build_vocab(train_iter, tokenizer_trg)

        # Preprocess data
        train_data_src, train_data_trg = loader.preprocess_data(train_iter, tokenizer_src, loader.source_vocab, max_len)
        valid_data_src, valid_data_trg = loader.preprocess_data(valid_iter, tokenizer_src, loader.source_vocab, max_len)
        test_data_src, test_data_trg = loader.preprocess_data(test_iter, tokenizer_src, loader.source_vocab, max_len)

        # Create DataLoaders
        train_loader = loader.create_dataloader(train_data_src, train_data_trg, batch_size)
        valid_loader = loader.create_dataloader(valid_data_src, valid_data_trg, batch_size)
        test_loader = loader.create_dataloader(test_data_src, test_data_trg, batch_size)
        return loader, train_loader, valid_loader, test_loader

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
