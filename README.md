# Transformer Implementation

This project implements a custom Transformer model from scratch and tests it on various datasets. We also compare its performance against PyTorch's `nn.Transformer`.

## Features
- Custom implementation of the Transformer architecture.
- Positional encoding, multi-head attention, and feedforward layers implemented manually.
- Training and evaluation on multiple datasets:
  - **WikiText-2**: A dataset for language modeling tasks.
  - **Multi30k (EN-DE)**: English to German translation.
  - **Multi30k (EN-FR)**: English to French translation.
- BLEU score and loss metrics for performance comparison.

## Comparisons
We evaluate our implementation against PyTorch's `nn.Transformer`. The comparison includes:
- Training time.
- Model accuracy (BLEU scores).
- Convergence behavior.

## Datasets
1. **WikiText-2**:
   - Used for language modeling.
   - Tokenized and preprocessed using `basic_english` tokenizer.

2. **Multi30k**:
   - Two settings: English-to-German (EN-DE) and English-to-French (EN-FR).
   - Preprocessed using `spacy` tokenizers for English, German, and French.

## Results
Results are evaluated based on BLEU scores and loss. Detailed analysis can be found in the logs and plots generated during training.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/username/transformer-project.git
2. Install dependencies:
   pip install -r requirements.txt
3. Run the training script:
   python train.py

Author: [Jonathan Wang, Conny Zhou]
