import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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

    def value_similarity(self, value1, value2, value_type):
        """
        Calculate similarity between two values using BLEU-4.

        Args:
            value1: First value (string or list of strings).
            value2: Second value (string or list of strings).
            value_type: Type of the value ("string" or "list").

        Returns:
            float: Similarity score between value1 and value2 (0 to 1).
        """
        # Direct match, return 1.0
        if value1 == value2:
            return 1.0

        smoothing_function = SmoothingFunction().method1

        if value_type == "string":
            # Convert strings to tokens
            value1_tokens = value1.split()
            value2_tokens = value2.split()
            return sentence_bleu([value1_tokens], value2_tokens, smoothing_function=smoothing_function)
        elif value_type == "list":
            # Combine list elements into a single string for comparison
            value1_combined = " ".join(value1)
            value2_combined = " ".join(value2)
            value1_tokens = value1_combined.split()
            value2_tokens = value2_combined.split()
            return sentence_bleu([value1_tokens], value2_tokens, smoothing_function=smoothing_function)
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


# Main function
if __name__ == "__main__":
    # Load dictionaries from JSON files
    with open("../attribute_definitions.json", "r") as f:
        attribute_definitions = json.load(f)

    with open("../value_types.json", "r") as f:
        value_types = json.load(f)

    # Instantiate the ObjectSimilarity class
    similarity_calculator = ObjectSimilarity(attribute_definitions, value_types)

    # Example objects
    obj1 = {
        "type": "attack-pattern",
        "name": "Spear",
        "description": " This individual name is xxx.",
        "kill_chain_phases": ["reconnaissance", "weaponization"]
    }

    obj2 = {
        "type": "attack-pattern",
        "name": "Spear phishing",
        "description": " This individual name is xxx.",
        "kill_chain_phases": ["reconnaissance", "weaponization"]
    }

    # Calculate similarity
    similarity = similarity_calculator.calculate_similarity(obj1, obj2)
    print(f"Similarity Score: {similarity:.2f}")
