import math
import multiprocessing
import time
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gc

from model.model.Transformer import Transformer
from utils.DataSummary import dataset_summary
from utils.loader_factory import get_loader_and_dataloaders
from utils.tools import idx_to_word, get_bleu


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=None):
        """
        Focal Loss for addressing class imbalance with an optional ignore index.
        Args:
            alpha (float): Balancing factor for each class.
            gamma (float): Focusing parameter to reduce loss for well-classified examples.
            reduction (str): Specifies the reduction type ('none', 'mean', 'sum').
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the loss.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes) containing model outputs.
            targets: Tensor of shape (batch_size,) containing ground truth labels.
        Returns:
            Computed Focal Loss value.
        """
        # Mask for valid targets (not equal to ignore_index)
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities

        # Create one-hot encoding of targets
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()

        # Compute probabilities of target classes
        probs_target = (probs * targets_one_hot).sum(dim=-1)

        # Compute focal weight
        focal_weight = (1 - probs_target) ** self.gamma

        # Compute the loss for valid targets
        loss = -self.alpha * focal_weight * torch.log(probs_target + 1e-8)
        loss = loss[valid_mask]  # Apply mask to exclude ignored indices

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=logits.device)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def repeated_token_loss(output, trg, vocab_size, alpha=1.0):
    """
    Computes a penalty for repeated tokens in the output compared to the target.

    Args:
        output: Tensor of shape (batch_size, seq_len, vocab_size), model output logits.
        trg: Tensor of shape (batch_size, seq_len), target token IDs.
        vocab_size: Size of the vocabulary.
        alpha: Scaling factor for the repetition penalty.

    Returns:
        Scalar tensor representing the repetition penalty loss.
    """
    batch_size, seq_len, vocab_size = output.shape

    # Get the predicted tokens (argmax over logits)
    predicted_tokens = output.argmax(dim=-1)  # (batch_size, seq_len)

    # Count occurrences of each token per sequence
    token_counts = torch.zeros(batch_size, vocab_size, device=output.device)
    for i in range(seq_len):
        token_counts.scatter_add_(1, predicted_tokens[:, i].unsqueeze(-1),
                                  torch.ones_like(predicted_tokens[:, i].unsqueeze(-1).float()))

    # Compute penalty for tokens appearing more than once
    repeated_penalty = torch.clamp(token_counts - 1, min=0)  # Penalize counts greater than 1
    loss = repeated_penalty.sum(dim=1).mean() * alpha  # Average over batch

    return loss


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.1, 0.1)





def prepare_data(batch, device):
    """
    Prepares source and target tensors by moving them to the specified device.
    """
    src = batch["input_ids"]  # Source sequences
    trg = batch["labels"]  # Target sequences

    # Move tensors to the specified device (e.g., GPU or CPU)
    src = src.to(device)
    trg = trg.to(device)

    return src, trg


# Training function
def train(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(dataloader):
        src, trg = prepare_data(batch, device=device)
        optimizer.zero_grad()
        output = model(src, trg[:, :-1], padding_idx=pad_token)
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        if i % 10 == 0:  # Log every 10 steps
            print(f"Step {i}/{len(dataloader)}: Loss = {loss.item():.4f}")
        if i % 100 == 0:
            # Decode and log predictions and targets
            print(f"\nIteration {i}/{len(dataloader)}:")
            print(f"Loss: {loss.item():.4f}")

            # Get the top predicted tokens
            predicted_tokens = output.argmax(dim=-1)
            for j in range(min(3, src.size(0))):  # Show up to 3 examples
                # Decode target and predicted sequences
                trg_sentence = idx_to_word(batch["labels"][j].tolist(), loader.reverse_target_vocab)
                pred_sentence = idx_to_word(predicted_tokens[j].tolist(), loader.reverse_target_vocab)
                print(f"Target Sentence  : {trg_sentence}")
                print(f"Predicted Sentence: {pred_sentence}")
                print()

    return epoch_loss / len(dataloader)


# Evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu_scores = []  # List to collect BLEU scores for all batches

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Prepare the data
            src, trg = prepare_data(batch, device=device)
            output = model(src, trg[:, :-1], padding_idx=pad_token)

            # Flatten outputs and targets for loss calculation
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_reshape = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output_reshape, trg_reshape)
            epoch_loss += loss.item()

            # BLEU calculation
            batch_bleu = []
            predicted_tokens = output.argmax(dim=-1)  # Get the top predicted tokens
            for j in range(len(batch["labels"])):  # Iterate over the batch
                # Decode target indices to words
                trg_sentence = idx_to_word(batch["labels"][j].tolist(), loader.reverse_target_vocab).split()

                # Decode predicted token indices to words
                pred_sentence = idx_to_word(predicted_tokens[j].tolist(), loader.reverse_target_vocab).split()

                # Compute BLEU score for this sentence pair
                bleu_score = get_bleu(hypotheses=pred_sentence, reference=trg_sentence)
                batch_bleu.append(bleu_score)

            # Compute average BLEU for this batch
            avg_batch_bleu = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0
            batch_bleu_scores.append(avg_batch_bleu)

    # Compute overall BLEU score across all batches
    overall_bleu = sum(batch_bleu_scores) / len(batch_bleu_scores) if batch_bleu_scores else 0
    return epoch_loss / len(dataloader), overall_bleu


# Test Function
def test_bleu(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu_scores = []  # List to collect BLEU scores for all batches

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Prepare the data
            src, trg = prepare_data(batch, device=device)
            output = model(src, trg[:, :-1], padding_idx=pad_token)

            # Flatten outputs and targets for loss calculation
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_reshape = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output_reshape, trg_reshape)
            epoch_loss += loss.item()

            # BLEU calculation
            batch_bleu = []
            predicted_tokens = output.argmax(dim=-1)  # Get the top predicted tokens
            for j in range(len(batch["labels"])):  # Iterate over the batch
                # Decode target indices to words
                trg_sentence = idx_to_word(batch["labels"][j].tolist(), loader.reverse_target_vocab).split()

                # Decode predicted token indices to words
                pred_sentence = idx_to_word(predicted_tokens[j].tolist(), loader.reverse_target_vocab).split()

                # Compute BLEU score for this sentence pair
                bleu_score = get_bleu(hypotheses=pred_sentence, reference=trg_sentence)
                batch_bleu.append(bleu_score)

            # Compute average BLEU for this batch
            avg_batch_bleu = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0
            batch_bleu_scores.append(avg_batch_bleu)

    # Compute overall BLEU score across all batches
    overall_bleu = sum(batch_bleu_scores) / len(batch_bleu_scores) if batch_bleu_scores else 0
    return epoch_loss / len(dataloader), overall_bleu


# Plot Loss Curve
def plot_loss_curve(train_losses, valid_losses, bleu_scores):
    epochs = range(1, len(train_losses) + 1)

    # Plotting Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, valid_losses, label="Validation Loss", marker='o')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting BLEU Scores
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, bleu_scores, label="BLEU Score", marker='o', color="green")
    plt.title("BLEU Score Curve")
    plt.xlabel("Epochs")
    plt.ylabel("BLEU Score")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    gc.collect()
    torch.cuda.empty_cache()

    # Main script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = "multi30k_en_de"
    batch_size = 64
    max_len = 256
    d_model = 512
    n_layers = 6
    n_heads = 8
    ffn_hidden = 2048
    drop_prob = 0.1
    init_lr = 1e-5
    warmup = 150
    weight_decay = 5e-4
    adam_eps = 5e-9
    total_epoch = 5

    print(f"Loading dataset: {dataset_name}...")
    dataset_name = "multi30k_en_de"  # Example: multi30k_en_de, multi30k_en_fr, wikitext
    batch_size = 64
    max_len = 256

    # Load dataset and DataLoaders
    loader, train_loader, valid_loader, test_loader = get_loader_and_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_len=max_len
    )

    # Summarize dataset
    dataset_summary(
        dataloaders={"train": train_loader, "validation": valid_loader, "test": test_loader},
        reverse_vocab_src=loader.reverse_source_vocab,
        reverse_vocab_trg=loader.reverse_target_vocab
    )

    enc_vocab_size = len(loader.source_vocab)
    dec_vocab_size = len(loader.target_vocab)
    pad_token = loader.source_vocab["<pad>"]


    # Model definition (assumes a Transformer model class exists)
    # model = TransformerModelBenchmark(
    #     src_vocab_size=enc_vocab_size,
    #     trg_vocab_size=dec_vocab_size,
    #     emb_dim=d_model,
    #     nhead=n_heads,
    #     num_encoder_layers=n_layers,
    #     num_decoder_layers=n_layers,
    #     dim_feedforward=ffn_hidden,
    #     dropout=drop_prob,
    #     max_len=max_len
    # ).to(device)

    # Define the model
    model = Transformer(
        src_vocab_size=len(loader.source_vocab),
        trg_vocab_size=len(loader.target_vocab),
        emb_dim=d_model,
        nhead=n_heads,
        num_encoder_layers=n_layers,
        num_decoder_layers=n_layers,
        dim_feedforward=ffn_hidden,
        drop_prob=drop_prob,
        max_len=max_len,
    ).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.apply(initialize_weights)

    # Define optimizer, scheduler, and criterion
    optimizer = Adam(
        params=model.parameters(),
        lr=init_lr,
        weight_decay=weight_decay,
        eps=adam_eps,
    )

    total_steps = total_epoch * len(train_loader)


    def warmup_cosine_lr(step):
        if step < warmup:
            return step / warmup
        else:
            progress = (step - warmup) / (total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))


    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)
    # criterion = FocalLoss(ignore_index=pad_token)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    train_losses, valid_losses, bleu_scores = [], [], []
    for epoch in range(total_epoch):
        train_loss = train(model, train_loader, optimizer, criterion, clip=1,
                           reverse_vocab_trg=loader.reverse_target_vocab)
        valid_loss, bleu = evaluate(model, valid_loader, criterion, loader.reverse_target_vocab)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        bleu_scores.append(bleu)

        # Save the best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"saved/model-{valid_loss:.4f}.pt")
        print(
            f"Epoch {epoch + 1}/{total_epoch}: Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, BLEU: {bleu:.4f}")

    print("\nTesting the model...")
    test_loss, test_bleu = evaluate(model, test_loader, criterion, loader.reverse_target_vocab)
    print(f"Test Loss: {test_loss:.4f}, Test BLEU: {test_bleu:.4f}")

    plot_loss_curve(train_losses, valid_losses, bleu_scores)
# Main Script with Test Phase
def run_with_test(total_epoch, best_loss=float("inf")):
    train_losses, valid_losses, bleu_scores = [], [], []

    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip=1)
        valid_loss, bleu = evaluate(model, valid_loader, criterion)
        end_time = time.time()

        # Log results
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        bleu_scores.append(bleu)

        # Save the best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"saved/model-{valid_loss:.4f}.pt")


        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f"Epoch: {step + 1} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}")
        print(f"\tVal Loss: {valid_loss:.3f} | Val PPL: {math.exp(valid_loss):.3f}")
        print(f"\tValidation BLEU Score: {bleu:.3f}")

    plot_loss_curve(train_losses, valid_losses, bleu_scores)

    # Test Phase
    print("\nTesting the model on the test set...")
    test_loss, test_bleu_score = test_bleu(model, test_loader, criterion)  # Avoid shadowing the function
    print(f"Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")
    print(f"Test BLEU Score: {test_bleu_score:.3f}")


