from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class AdvancedBLEUSimilarity:
    def __init__(self, attribute_definitions, value_types):
        """
        Initialize the AdvancedBLEUSimilarity class with attribute definitions and value types.

        Args:
            attribute_definitions (dict): Dictionary defining required and optional attributes for each object type.
            value_types (dict): Dictionary defining the type of each attribute (e.g., "string" or "list").
        """
        self.attribute_definitions = attribute_definitions
        self.value_types = value_types

    def value_similarity_BLEU(self, value1, value2, value_type):
        """
        Calculate similarity between two values using BLEU-4 score.
        """
        if value1 == value2:
            print(f"[BLEU] Values are identical: {value1} == {value2}, similarity = 1.0")
            return 1.0

        smoothing_function = SmoothingFunction().method1
        if value_type == "string":
            value1_tokens = value1.split()
            value2_tokens = value2.split()
            score = sentence_bleu([value1_tokens], value2_tokens, smoothing_function=smoothing_function)
            print(f"[BLEU] String similarity: {value1} <-> {value2}, score = {score:.4f}")
            return score
        elif value_type == "list":
            value1_combined = " ".join(value1)
            value2_combined = " ".join(value2)
            value1_tokens = value1_combined.split()
            value2_tokens = value2_combined.split()
            score = sentence_bleu([value1_tokens], value2_tokens, smoothing_function=smoothing_function)
            print(f"[BLEU] List similarity: {value1_combined} <-> {value2_combined}, score = {score:.4f}")
            return score
        else:
            raise ValueError(f"Unsupported value type: {value_type}")

    def advanced_similarity(self, obj1, obj2, object_type, w_type=0.5, v=2):
        """
        Advanced similarity calculation using BLEU score with dynamic weight distribution.
        """
        print("\n--- Advanced BLEU Similarity Calculation ---")
        print(f"Comparing objects: {obj1} <-> {obj2}")

        # Step 1: Type similarity
        p_type = 1 if obj1.get("type") == obj2.get("type") else 0
        s_type = p_type  # 类型完全相同直接得分为 1.0
        print(f"Type similarity: p_type = {p_type}, weighted = {s_type:.4f}")

        # Step 2: Attribute similarity
        attributes = self.attribute_definitions.get(object_type, {})
        required_attrs = attributes.get("required_attributes", [])
        optional_attrs = attributes.get("optional_attributes", [])

        total_weight = len(required_attrs) * v + len(optional_attrs)
        w_ra = v / total_weight
        w_oa = 1 / total_weight
        print(f"Weighting: w_ra (required) = {w_ra:.4f}, w_oa (optional) = {w_oa:.4f}")

        # Calculate required and optional similarities
        s_required = self._calculate_attribute_similarity(
            obj1, obj2, required_attrs, w_ra, self.value_similarity_BLEU
        )
        s_optional = self._calculate_attribute_similarity(
            obj1, obj2, optional_attrs, w_oa, self.value_similarity_BLEU
        )
        print(f"Attribute similarities: s_required = {s_required:.4f}, s_optional = {s_optional:.4f}")

        # Total similarity with normalization
        s_total = (s_type + s_required + s_optional) / (1 + 1 - w_type)
        print(f"Total similarity: s_total = {s_total:.4f}")
        print("--- End of Calculation ---\n")
        return s_total

    def _calculate_attribute_similarity(self, obj1, obj2, attributes, weight, similarity_func):
        shared_attributes = [attr for attr in attributes if attr in obj1 and attr in obj2]
        print(f"Shared attributes: {shared_attributes}")

        if not shared_attributes:
            print(f"No shared attributes found, returning 0.0")
            return 0.0

        score = 0.0
        for attr in shared_attributes:
            attr_similarity = similarity_func(obj1[attr], obj2[attr], self.value_types.get(attr, "string"))
            weighted_similarity = (weight / len(shared_attributes)) * attr_similarity
            score += weighted_similarity
            print(f"Attribute: {attr}, similarity = {attr_similarity:.4f}, weighted = {weighted_similarity:.4f}")

        return score


# Example usage
if __name__ == "__main__":
    # Example attribute definitions and value types
    attribute_definitions = {
        "attack-pattern": {
            "required_attributes": ["name", "description"],
            "optional_attributes": ["kill_chain_phases"]
        }
    }

    value_types = {
        "name": "string",
        "description": "string",
        "kill_chain_phases": "list"
    }

    # Example objects
    obj1 = {
        "type": "attack-pattern",
        "name": "Spear ",
        "description": "Phishing attack targeting specific users.",
        "kill_chain_phases": ["weaponization"]
    }

    obj2 = {
        "type": "attack-pattern",
        "name": "Spear Phishing",
        "description": "Phishing attack targeting specific users.",
        "kill_chain_phases": ["reconnaissance", "weaponization"]
    }

    # Instantiate and calculate similarity
    similarity_calculator = AdvancedBLEUSimilarity(attribute_definitions, value_types)
    similarity_score = similarity_calculator.advanced_similarity(obj1, obj2, "attack-pattern", w_type=0.5, v=2)

    print(f"Advanced BLEU Similarity Score: {similarity_score:.4f}")
