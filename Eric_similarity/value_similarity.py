from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

def value_similarity(input1, input2, model_name='all-MiniLM-L6-v2'):
    """
    Calculate similarity between two values (strings or lists of strings) using SentenceTransformer embeddings.

    Args:
        input1: str or list of str - First input (string or list of strings).
        input2: str or list of str - Second input (string or list of strings).
        model_name: str - Name of the SentenceTransformer model to use for embeddings.

    Returns:
        float: Similarity score between input1 and input2 (0 to 1).
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer(model_name)

    # Check if inputs are strings
    if isinstance(input1, str) and isinstance(input2, str):
        # Compute embeddings for both strings
        embedding1 = model.encode(input1)
        embedding2 = model.encode(input2)

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]

    # Check if inputs are lists
    elif isinstance(input1, list) and isinstance(input2, list):
        # Compute embeddings for all elements in the lists
        embeddings1 = [model.encode(item) for item in input1]
        embeddings2 = [model.encode(item) for item in input2]

        # Calculate the average embeddings for each list
        avg_embedding1 = np.mean(embeddings1, axis=0)
        avg_embedding2 = np.mean(embeddings2, axis=0)

        # Calculate cosine similarity between the average embeddings
        similarity = cosine_similarity([avg_embedding1], [avg_embedding2])[0][0]

    else:
        raise ValueError("Inputs must both be strings or both be lists of strings.")

    return similarity

# Example usage
if __name__ == "__main__":
    # For strings
    text1 = "This is not a similar text"
    text2 = "This is a similar text"
    print("String Similarity:", value_similarity(text1, text2))

    # For lists
    list1 = ["apple", "banana", "fruit"]
    list2 = ["apple", "banana", "fruits"]
    print("List Similarity:", value_similarity(list1, list2))
