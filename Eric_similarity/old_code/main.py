import json
from ObjectSimilarity import ObjectSimilarity


def main():
    # Load the attribute dictionary from an external JSON file
    with open("object_classification.json", "r") as file:
        attribute_dict = json.load(file)

    # Initialize the similarity calculator
    similarity_calculator = ObjectSimilarity(attribute_dict)

    # Define two STIX objects for comparison
    stix_object1 = {
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

    stix_object2 = {
    "type": "threat-actor",
    "id": "threat-actor--1",
    "name": "Disco Team Threat Actor Group",
    "description": "This organized threat actor group operates to create profit from all types of crime.",
    "aliases": ["Equipo del Discoteca"],
    "goals": ["Acquire Credit Card Data"],
    "sophistication": "expert",
    "resource_level": "organization",
    "primary_motivation": "financial-gain"
  }

    # Calculate similarity
    similarity_score = similarity_calculator.calculate_similarity(stix_object1, stix_object2)
    print(f"Similarity Score: {similarity_score}")


if __name__ == "__main__":
    main()
