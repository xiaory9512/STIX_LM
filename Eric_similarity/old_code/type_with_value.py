import json
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to check if the type attribute matches and assign the corresponding score.
def match_object_type(obj1, obj2):
    return 0.5 if obj1.get("type") == obj2.get("type") else 0.0

# Function to compare attribute types and calculate similarity score
def match_attribute_type(obj1, obj2, attribute_weights):
    obj_type = obj1.get("type")

    if obj_type not in attribute_weights:
        return 0.0

    weights = attribute_weights[obj_type]
    required_attrs = weights["required"]
    optional_attrs = weights["optional"]
    score = 0.0
    total_possible_score = 0.0  # 用於計算所有實際存在的屬性總權重

    # Check required attributes
    for attr, weight in required_attrs.items():
        if attr in obj1 and attr in obj2:
            total_possible_score += weight  # 累加權重
            if obj1.get(attr) == obj2.get(attr):
                score += weight

    # Check optional attributes and calculate semantic similarity where applicable
    excluded_attrs = {"id", "spec_version", "type"}
    all_attrs = set(obj1.keys()).union(obj2.keys()) - set(required_attrs.keys()) - excluded_attrs

    for attr in all_attrs:
        if attr in optional_attrs:
            weight = optional_attrs[attr]
            total_possible_score += weight  # 累加權重
            attr_value1, attr_value2 = obj1.get(attr), obj2.get(attr)
            if isinstance(attr_value1, str) and isinstance(attr_value2, str):
                # Calculate semantic similarity for string attributes
                similarity = calculate_semantic_similarity(attr_value1, attr_value2)
                score += min(similarity * weight, weight)  # 確保不超過該屬性的最大權重
            elif attr_value1 == attr_value2:
                score += weight

    # Normalize the score by the total possible score
    return score / total_possible_score if total_possible_score > 0 else 0.0




# Function to calculate semantic similarity between two attribute values using embeddings
def calculate_semantic_similarity(value1, value2):
    # Check if the two attribute values are exactly the same
    if value1 == value2:
        return 1.0  # 100% similarity if values are identical
    # Encode the attribute values into embeddings
    embedding1 = model.encode(value1, convert_to_tensor=True)
    embedding2 = model.encode(value2, convert_to_tensor=True)

    # Compute cosine similarity between the embeddings
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity

# Combine the two scores into a final similarity score
def calculate_similarity_score(obj1, obj2, attribute_weights):
    type_score = match_object_type(obj1, obj2)
    attribute_score = match_attribute_type(obj1, obj2, attribute_weights)
    return type_score + attribute_score

# Load attribute weights from a JSON file
def load_attribute_weights(filepath):
    with open(filepath, "r") as file:
        return json.load(file)

# Main function for testing
def main():
    attribute_weights = load_attribute_weights("attribute_weights.json")

    # Sample objects for testing
    obj1 = {
        "type": "threat-actor",
        "id": "threat-actor--1",
        "name": "Disco Team Threat Actor Group",
        "description": "This organized threat actor group operates to create profit from all types of crime.",
        "aliases": ["Equipo del Discoteca"],
        "goals": ["Steal Credit Card Information"],
        "sophistication": "expert",
        "resource_level": "organization",
        "primary_motivation": "finicial-gain"
  }

    obj2 = {
        "type": "threat-actor",
        "id": "threat-actor--1",
        "name": "Disco Team Threat Actor Group",
        "description": "This organized threat actor group operates to create profit from all types of crime.",
        "aliases": ["Equipo del Discoteca"],
        "goals": ["Steal Credit Card Information"],
        "sophistication": "expert"
  }


    similarity_score = calculate_similarity_score(obj1, obj2, attribute_weights)
    print(f"Similarity Score: {similarity_score}")

if __name__ == "__main__":
    main()
