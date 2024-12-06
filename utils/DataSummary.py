from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def dataset_summary(dataloaders, reverse_vocab_src, reverse_vocab_trg):
    """
    Summarizes and describes the dataset based on the DataLoader.
    Args:
        dataloaders: A dictionary of PyTorch DataLoaders for train, validation, and test datasets.
        reverse_vocab_src: Reverse vocabulary for source language.
        reverse_vocab_trg: Reverse vocabulary for target language.
    """
    all_src_lengths = []
    all_trg_lengths = []
    src_token_counts = Counter()
    trg_token_counts = Counter()

    print("Analyzing the dataset...")
    for split_name, dataloader in dataloaders.items():
        print(f"\nProcessing {split_name} dataset...")
        for batch in dataloader:
            src, trg = batch["input_ids"], batch["labels"]

            # Calculate sentence lengths
            all_src_lengths.extend([len(seq[seq != 0]) for seq in src])  # Exclude padding tokens
            all_trg_lengths.extend([len(seq[seq != 0]) for seq in trg])  # Exclude padding tokens

            # Collect token counts
            for seq in src:
                src_token_counts.update([reverse_vocab_src.get(tok.item(), "<unk>") for tok in seq if tok.item() in reverse_vocab_src])
            for seq in trg:
                trg_token_counts.update([reverse_vocab_trg.get(tok.item(), "<unk>") for tok in seq if tok.item() in reverse_vocab_trg])

    # Source language statistics
    print("\nSource Language Statistics:")
    print(f"Average Sentence Length: {np.mean(all_src_lengths):.2f}")
    print(f"Median Sentence Length: {np.median(all_src_lengths):.2f}")
    print(f"Max Sentence Length: {np.max(all_src_lengths)}")
    print(f"Min Sentence Length: {np.min(all_src_lengths)}")
    print(f"Vocabulary Size: {len(reverse_vocab_src)}")
    print(f"Most Common Tokens: {src_token_counts.most_common(10)}")

    # Target language statistics
    print("\nTarget Language Statistics:")
    print(f"Average Sentence Length: {np.mean(all_trg_lengths):.2f}")
    print(f"Median Sentence Length: {np.median(all_trg_lengths):.2f}")
    print(f"Max Sentence Length: {np.max(all_trg_lengths)}")
    print(f"Min Sentence Length: {np.min(all_trg_lengths)}")
    print(f"Vocabulary Size: {len(reverse_vocab_trg)}")
    print(f"Most Common Tokens: {trg_token_counts.most_common(10)}")

    # Plot Sentence Length Distribution
    def plot_length_distribution(lengths, title):
        plt.figure(figsize=(8, 5))
        plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel("Sentence Length")
        plt.ylabel("Frequency")
        plt.show()

    plot_length_distribution(all_src_lengths, "Source Sentence Length Distribution")
    plot_length_distribution(all_trg_lengths, "Target Sentence Length Distribution")

#
# # Example usage for entire dataset
# dataloaders = {
#     "train": train_loader,
#     "validation": valid_loader,
#     "test": test_loader
# }
#
# dataset_summary(dataloaders, loader.reverse_source_vocab, loader.reverse_target_vocab)
