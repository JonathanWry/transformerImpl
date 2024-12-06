import sacrebleu

def get_bleu(hypotheses, reference):
    """
    Compute BLEU score for a single hypothesis and reference pair using sacrebleu.
    Args:
        hypotheses: List of predicted tokens (as words).
        reference: List of ground truth tokens (as words).
    Returns:
        BLEU score (float).
    """
    # SacreBLEU expects references as a list of lists and hypothesis as a single string
    reference_str = " ".join(reference)
    hypothesis_str = " ".join(hypotheses)

    bleu = sacrebleu.sentence_bleu(hypothesis_str, [reference_str])
    return bleu.score


# Perplexity Calculation Function
def calculate_perplexity(loss):
    """
    Calculate perplexity from loss.
    Args:
        loss: Loss value (float).
    Returns:
        Perplexity (float).
    """
    return math.exp(loss) if loss < float('inf') else float('inf')

def idx_to_word(token_ids, reverse_vocab):
    """
    Converts a list of token IDs or a single token ID back into a sentence string using the reverse vocabulary.

    Args:
        token_ids (list of int or int): Token IDs to be converted.
        reverse_vocab (dict): Mapping from token IDs to words.

    Returns:
        str: The decoded sentence as a string.
    """
    # Ensure token_ids is a list
    if isinstance(token_ids, int):
        token_ids = [token_ids]

    # Use reverse_vocab to map token IDs to words
    tokens = [reverse_vocab[token_id] for token_id in token_ids if token_id in reverse_vocab]

    # Join tokens into a sentence, skipping special tokens like <pad>, <bos>, <eos>
    sentence = " ".join([token for token in tokens if not token.startswith("<")])
    return sentence
