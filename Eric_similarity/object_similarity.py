import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
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
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def value_similarity_COS(self, value1, value2, value_type):
        """
        Calculate similarity between two values using cosine similarity.
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

    def value_similarity_BLEU(self, value1, value2, value_type):
        """
        Calculate similarity between two values using BLEU-4 score.
        """
        if value1 == value2:
            return 1.0

        smoothing_function = SmoothingFunction().method1
        if value_type == "string":
            value1_tokens = value1.split()
            value2_tokens = value2.split()
            return sentence_bleu([value1_tokens], value2_tokens, smoothing_function=smoothing_function)
        elif value_type == "list":
            value1_combined = " ".join(value1)
            value2_combined = " ".join(value2)
            value1_tokens = value1_combined.split()
            value2_tokens = value2_combined.split()
            return sentence_bleu([value1_tokens], value2_tokens, smoothing_function=smoothing_function)
        else:
            raise ValueError(f"Unsupported value type: {value_type}")

    def basic_similarity(self, obj1, obj2, method="COS"):
        """
        Calculate similarity using a basic weighted scheme for required and optional attributes.
        """
        value_similarity = (
            self.value_similarity_COS if method == "COS" else self.value_similarity_BLEU
        )

        type1, type2 = obj1.get("type"), obj2.get("type")
        if type1 != type2:
            return 0.0

        attributes = self.attribute_definitions.get(type1, {})
        required_attrs = attributes.get("required_attributes", [])
        optional_attrs = attributes.get("optional_attributes", [])

        total_score = 0.5  # Type similarity
        required_score = self._calculate_attribute_similarity(
            obj1, obj2, required_attrs, 0.3, value_similarity
        )
        optional_score = self._calculate_attribute_similarity(
            obj1, obj2, optional_attrs, 0.2, value_similarity
        )

        return total_score + required_score + optional_score

    #
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
        if not shared_attributes:
            return 0.0

        score = sum(
            (weight / len(shared_attributes)) *
            similarity_func(obj1[attr], obj2[attr], self.value_types.get(attr, "string"))
            for attr in shared_attributes
        )
        return score
