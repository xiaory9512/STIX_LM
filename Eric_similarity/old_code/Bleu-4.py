from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


# Function to calculate BLEU-4 score for two sentences
def calculate_bleu(reference: str, hypothesis: str) -> float:
    """
    Calculate the BLEU-4 score between a reference sentence and a hypothesis.
    """
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()

    # Smoothing function to handle edge cases like empty hypothesis
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing)

    return bleu_score


# Function to calculate BLEU-4 scores for two lists of sentences
def calculate_bleu_for_lists(references: list, hypotheses: list) -> list:
    """
    Calculate BLEU-4 scores for two lists of sentences.
    Assumes both lists are of the same length.
    """
    if len(references) != len(hypotheses):
        raise ValueError("The two lists must have the same length.")

    scores = []
    for ref, hyp in zip(references, hypotheses):
        score = calculate_bleu(ref, hyp)
        scores.append(score)

    return scores


# Example usage
if __name__ == "__main__":
    # Single pair of sentences
    reference_sentence = "The cat is on the mat."
    hypothesis_sentence = "The cat is on the mat."
    print("BLEU-4 score for single pair:", calculate_bleu(reference_sentence, hypothesis_sentence))

    # Lists of sentences
    reference_list = ["Steal Credit Card Information"]
    hypothesis_list = [
        "The cat sits on the mat.",
        "A dog is in the house."
    ]
    scores = calculate_bleu_for_lists(reference_list, hypothesis_list)
    print("BLEU-4 scores for lists:", scores)
