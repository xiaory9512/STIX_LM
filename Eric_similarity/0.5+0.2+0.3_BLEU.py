import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

class ObjectSimilarityFixedBLEU:
    def __init__(self, attribute_definitions, value_types):
        self.attribute_definitions = attribute_definitions
        self.value_types = value_types

    def value_similarity(self, value1, value2, value_type):
        if value1 == value2:
            return 1.0
        smoothing_function = SmoothingFunction().method1
        if value_type == "string":
            return sentence_bleu([value1.split()], value2.split(), smoothing_function=smoothing_function)
        elif value_type == "list":
            value1_combined = " ".join(value1)
            value2_combined = " ".join(value2)
            return sentence_bleu([value1_combined.split()], value2_combined.split(), smoothing_function=smoothing_function)
        else:
            raise ValueError("Unsupported value type.")

    def calculate_similarity(self, obj1, obj2):
        type1, type2 = obj1.get("type"), obj2.get("type")
        if type1 != type2:
            return 0.0
        total_score = 0.5
        attributes = self.attribute_definitions.get(type1, {})
        required_attrs = attributes.get("required_attributes", [])
        optional_attrs = attributes.get("optional_attributes", [])

        # Required attributes
        shared_required = [attr for attr in required_attrs if attr in obj1 and attr in obj2]
        if shared_required:
            total_score += 0.3 * sum(self.value_similarity(obj1[attr], obj2[attr], self.value_types[attr]) for attr in shared_required) / len(shared_required)

        # Optional attributes
        shared_optional = [attr for attr in optional_attrs if attr in obj1 and attr in obj2]
        if shared_optional:
            total_score += 0.2 * sum(self.value_similarity(obj1[attr], obj2[attr], self.value_types[attr]) for attr in shared_optional) / len(shared_optional)

        return total_score

if __name__ == "__main__":
    # Load attribute definitions and value types from JSON files
    with open("attribute_definitions.json", "r") as f:
        attribute_definitions = json.load(f)

    with open("value_types.json", "r") as f:
        value_types = json.load(f)

    # Instantiate the ObjectSimilarity class
    similarity_calculator = ObjectSimilarityFixedBLEU(attribute_definitions, value_types)

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
