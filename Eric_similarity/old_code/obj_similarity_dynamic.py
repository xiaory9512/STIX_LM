from nltk.translate.bleu_score import sentence_bleu
import json


class ObjectSimilarity:
    def __init__(self, value_types_file, attribute_definitions_file):
        """
        初始化 ObjectSimilarity 类，加载 value types 和 attribute definitions。
        :param value_types_file: 包含属性类型定义的 JSON 文件路径
        :param attribute_definitions_file: 包含属性分类的 JSON 文件路径
        """
        with open(value_types_file, 'r') as f:
            self.value_types = json.load(f)

        with open(attribute_definitions_file, 'r') as f:
            self.attribute_definitions = json.load(f)

    def compute_similarity(self, obj1, obj2, object_type, w_type=0.5, v=2):
        """
        计算两个对象之间的相似度。
        :param obj1: 第一个对象
        :param obj2: 第二个对象
        :param object_type: 对象类型，例如 "attack-pattern" 或 "campaign"
        :param w_type: 类型相似度的权重
        :param v: 调整 required 和 optional 属性权重比例的系数
        :return: 相似度分数
        """
        # Step 1: Type similarity
        p_type = 1 if obj1.get("type") == obj2.get("type") else 0
        s_type = w_type * p_type

        # Step 2: Attribute similarity
        attributes = self.attribute_definitions.get(object_type, {})
        required_attributes = attributes.get("required_attributes", [])
        optional_attributes = attributes.get("optional_attributes", [])

        # Compute weights
        m = len(required_attributes)
        n = len(optional_attributes)
        w_oa = (1 - w_type) / (m * v + n)
        w_ra = v * w_oa

        # Calculate required attributes similarity
        s_required = self._calculate_attribute_similarity(
            obj1, obj2, required_attributes, w_ra
        )

        # Calculate optional attributes similarity
        s_optional = self._calculate_attribute_similarity(
            obj1, obj2, optional_attributes, w_oa
        )

        # Total similarity
        s_total = s_type + (1 - w_type) * (s_required + s_optional)
        return s_total

    def _calculate_attribute_similarity(self, obj1, obj2, attributes, weight):
        """
        计算属性相似度。
        :param obj1: 第一个对象
        :param obj2: 第二个对象
        :param attributes: 要计算的属性列表
        :param weight: 单个属性的权重
        :return: 属性相似度分数
        """
        shared_attributes = [
            attr for attr in attributes if attr in obj1 and attr in obj2
        ]
        if not shared_attributes:
            return 0

        similarity = 0
        for attr in shared_attributes:
            value1 = obj1[attr]
            value2 = obj2[attr]
            value_similarity = self._compute_value_similarity(value1, value2)
            similarity += weight * value_similarity
        return similarity

    def _compute_value_similarity(self, value1, value2):
        """
        使用 BLEU score 计算两个值的相似度。
        :param value1: 第一个值（可以是 string 或 list）
        :param value2: 第二个值（可以是 string 或 list）
        :return: 值的相似度分数
        """
        if isinstance(value1, str) and isinstance(value2, str):
            return sentence_bleu([value1.split()], value2.split())
        elif isinstance(value1, list) and isinstance(value2, list):
            # Handle list similarity
            total_similarity = 0
            for item1 in value1:
                max_similarity = max(
                    sentence_bleu([item2.split()], item1.split()) for item2 in value2
                )
                total_similarity += max_similarity
            return total_similarity / max(len(value1), len(value2))
        else:
            return 0  # Different types, no similarity


# 使用示例
# 创建实例
similarity_calculator = ObjectSimilarity("../value_types.json", "attribute_definitions.json")

# 定义两个对象
object1 = {
    "type": "attack-pattern",
    "name": "Phishing",
    "description": "A phishing attack.",
    "aliases": ["Email Fraud"],
}

object2 = {
    "type": "attack-pattern",
    "name": "Phishing",
    "description": "Phishing attacks.",
    "aliases": ["Email Fraud"],
}

# 计算相似度
score = similarity_calculator.compute_similarity(object1, object2, "attack-pattern")
print(f"Similarity Score: {score}")
