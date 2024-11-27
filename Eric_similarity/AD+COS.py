import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

class ObjectSimilarityAdvancedCosine:
    def __init__(self, attribute_definitions, value_types, type_weight=0.5):
        self.attribute_definitions = attribute_definitions
        self.value_types = value_types
        self.type_weight = type_weight
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def value_similarity(self, value1, value2, value_type):
        if isinstance(value1, str):
            vec1 = self.model.encode(value1)
            vec2 = self.model.encode(value2)
        elif isinstance(value1, list):
            vec1 = np.mean([self.model.encode(v) for v in value1], axis=0)
            vec2 = np.mean([self.model.encode(v) for v in value2], axis=0)
        else:
            raise ValueError("Unsupported value type.")
        return cosine_similarity([vec1], [vec2])[0][0]

    def calculate_similarity(self, obj1, obj2):
        type1, type2 = obj1.get("type"), obj2.get("type")
        if type1 != type2:
            return 0.0

        total_score = 0.0
        attributes = self.attribute_definitions.get(type1, {})
        required_attrs = attributes.get("required_attributes", [])
        optional_attrs = attributes.get("optional_attributes", [])

        # Calculate required attribute similarity

        # missing pairing attribute algorithms, 3 scenarios.
        shared_required = [attr for attr in required_attrs if attr in obj1 and attr in obj2]
        if shared_required:
            weight_per_attr = self.type_weight / len(shared_required)  # Dynamic distribution of 0.5 weight
            for attr in shared_required:
                similarity = self.value_similarity(obj1[attr], obj2[attr], self.value_types[attr])
                total_score += weight_per_attr * similarity

        # Calculate optional attribute similarity
        shared_optional = [attr for attr in optional_attrs if attr in obj1 and attr in obj2]
        if shared_optional:
            weight_per_attr = self.type_weight / len(shared_optional)  # Dynamic distribution of 0.5 weight
            for attr in shared_optional:
                similarity = self.value_similarity(obj1[attr], obj2[attr], self.value_types[attr])
                total_score += weight_per_attr * similarity

        return total_score

if __name__ == "__main__":
    # Load attribute definitions and value types from JSON files
    with open("attribute_definitions.json", "r") as f:
        attribute_definitions = json.load(f)

    with open("value_types.json", "r") as f:
        value_types = json.load(f)

    # Instantiate the ObjectSimilarity class
    similarity_calculator = ObjectSimilarityAdvancedCosine(attribute_definitions, value_types)

    # Folder containing JSON objects
    folder_path = "./testcase/object_level/threat-actor1"  # Replace with the path to your folder

    # Read the primary object from "sample_o.json"
    object1_path = os.path.join(folder_path, "sample_0.json")
    with open(object1_path, "r") as f:
        object1 = json.load(f)

    # Iterate through other JSON files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith("sample") and filename.endswith(".json") and filename != "sample_o.json":
            object2_path = os.path.join(folder_path, filename)
            with open(object2_path, "r") as f:
                object2 = json.load(f)

            # Calculate similarity
            similarity = similarity_calculator.calculate_similarity(object1, object2)

            # Print result
            print(f"Object1: sample_0.json, Object2: {filename}, Similarity: {similarity:.2f}")
