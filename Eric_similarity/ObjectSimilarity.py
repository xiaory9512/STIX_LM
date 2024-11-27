import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json


class ObjectSimilarity:
    def __init__(self, attribute_definitions, value_types):
        """
        Initialize the ObjectSimilarity class with attribute definitions and value types.

        Args:
            attribute_definitions (dict): Dictionary defining required and optional attributes for each object type.
            value_types (dict): Dictionary defining the type of each attribute (e.g., "string" or "list").
        """
        self.attribute_definitions = attribute_definitions
        self.value_types = value_types
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def value_similarity(self, value1, value2, value_type):
        """
        Calculate similarity between two values (strings or lists of strings) using SentenceTransformer embeddings.

        Args:
            value1: First value (string or list of strings).
            value2: Second value (string or list of strings).
            value_type: Type of the value ("string" or "list").

        Returns:
            float: Similarity score between value1 and value2 (0 to 1).
        """
        if value_type == "string":
            embedding1 = self.model.encode(value1)
            embedding2 = self.model.encode(value2)
            return cosine_similarity([embedding1], [embedding2])[0][0]
        elif value_type == "list":
            embeddings1 = [self.model.encode(item) for item in value1]
            embeddings2 = [self.model.encode(item) for item in value2]
            avg_embedding1 = np.mean(embeddings1, axis=0)
            avg_embedding2 = np.mean(embeddings2, axis=0)
            return cosine_similarity([avg_embedding1], [avg_embedding2])[0][0]
        else:
            raise ValueError(f"Unsupported value type: {value_type}")

    def calculate_similarity(self, obj1, obj2):
        """
        Calculate the similarity between two STIX objects.

        Args:
            obj1 (dict): First STIX object.
            obj2 (dict): Second STIX object.

        Returns:
            float: Similarity score between obj1 and obj2 (0 to 1).
        """
        # Validate object types
        type1, type2 = obj1.get("type"), obj2.get("type")
        if type1 != type2:
            return 0.0

        type_weight = 0.5
        required_weight = 0.3
        optional_weight = 0.2
        total_score = 0.5  # Start with type similarity as 0.5 since types are the same

        # Get attributes for the type
        attributes = self.attribute_definitions.get(type1, {})
        required_attrs = attributes.get("required_attributes", [])
        optional_attrs = attributes.get("optional_attributes", [])

        # Calculate required attribute similarity
        shared_required = [attr for attr in required_attrs if attr in obj1 and attr in obj2]
        if shared_required:
            m = len(shared_required)
            required_score = sum(
                (required_weight / m) *
                self.value_similarity(
                    obj1[attr], obj2[attr], self.value_types.get(attr, "string")
                )
                for attr in shared_required
            )
            total_score += required_score

        # Calculate optional attribute similarity
        shared_optional = [attr for attr in optional_attrs if attr in obj1 and attr in obj2]
        if shared_optional:
            n = len(shared_optional)
            optional_score = sum(
                (optional_weight / n) *
                self.value_similarity(
                    obj1[attr], obj2[attr], self.value_types.get(attr, "string")
                )
                for attr in shared_optional
            )
            total_score += optional_score

        return total_score

if __name__ == "__main__":
    # Load dictionaries from JSON files
    with open("./attribute_definitions.json", "r") as f:
        attribute_definitions = json.load(f)

    with open("./value_types.json", "r") as f:
        value_types = json.load(f)

    # Instantiate the ObjectSimilarity class
    similarity_calculator = ObjectSimilarity(attribute_definitions, value_types)

    # Example objects
    obj1 = {
    "type": "threat-actor",
    "id": "threat-actor--1",
    "name": "Disco Team Threat Actor Group",
    "description": "This organized threat actor group operates to create profit from all types of crime.",
    "aliases": [
        "Equipo del Discoteca"
    ],
    "threat_actor_types": [
        "financial-crime"
    ],
    "goals": [
        "Steal Credit Card Information",
        "Launder Money"
    ],
    "sophistication": "expert",
    "resource_level": "organization",
    "primary_motivation": "financial-gain",
    "secondary_motivations": [
        "personal-satisfaction"
    ],
    "roles": [
        "operator",
        "planner"
    ],
    "first_seen": "2023-01-01T00:00:00Z",
    "last_seen": "2024-01-01T00:00:00Z"
}

    obj2 = {
    "type": "threat-actor",
    "id": "threat-actor--1",
    "name": "Disco Team Threat Actor Group",
    "description": "This organized threat actor group operates to create profit from all types of crime.",
    "aliases": [
        "Equipo del Discoteca"
    ],
    "threat_actor_types": [
        "financial-crime"
    ],
    "goals": [
        "Steal Credit Card Information",
        "Launder Money"
    ],
    "sophistication": "novice",
    "resource_level": "organization",
    "primary_motivation": "ideological",
    "secondary_motivations": [
        "personal-satisfaction"
    ],
    "roles": [
        "hacker",
        "spammer"
    ],
    "first_seen": "2023-01-01T00:00:00Z",
    "last_seen": "2024-01-01T00:00:00Z"
}
    # Calculate similarity
    similarity = similarity_calculator.calculate_similarity(obj1, obj2)
    print(f"Similarity Score: {similarity:.2f}")

