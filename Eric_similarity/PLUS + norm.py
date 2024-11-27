import os
import json
from nltk.translate.bleu_score import sentence_bleu
from sklearn.preprocessing import MinMaxScaler


class ObjectSimilarityPlusMethod:
    def __init__(self, attribute_definitions, value_types):
        self.attribute_definitions = attribute_definitions
        self.value_types = value_types

    def calculate_similarity(self, obj1, obj2, threshold=0.85):
        # 初始化分数
        score = 0

        # 确保类型匹配
        if obj1['type'] == obj2['type']:
            score += 100
        else:
            return 0  # 如果类型不匹配，相似度直接为0

        # 获取必需和可选属性
        attributes = self.attribute_definitions.get(obj1['type'], {})
        required_attrs = attributes.get("required_attributes", [])
        optional_attrs = attributes.get("optional_attributes", [])

        # 计算分数
        def compare_attributes(attr_dict1, attr_dict2, weight):
            local_score = 0
        # pariring first using attribute name to pair each line of attribute
            for attr in attr_dict1:
                if attr in attr_dict2:
                    # 使用 BLEU 分数检查相似性
                    bleu_score = sentence_bleu([attr_dict2[attr].split()], attr_dict1[attr].split())
                    if bleu_score >= threshold:
                        local_score += weight
                    else:
                        local_score -= weight
                else:
                    local_score -= weight  # 如果 obj2 缺失 obj1 的属性
            for attr in attr_dict2:
                if attr not in attr_dict1:
                    local_score -= weight  # 如果 obj2 多出 obj1 没有的属性
            return local_score

        # 对比必需属性
        score += compare_attributes(obj1.get('required_attributes', {}), obj2.get('required_attributes', {}), 30)

        # 对比可选属性
        score += compare_attributes(obj1.get('optional_attributes', {}), obj2.get('optional_attributes', {}), 15)

        # 将得分缩放至 [0, 1]
        # 假设最大可能分数为 100 + 30 * len(required_attrs) + 15 * len(optional_attrs)
        max_score = 100 + 30 * len(required_attrs) + 15 * len(optional_attrs)
        min_score = -30 * len(required_attrs) - 15 * len(optional_attrs)

        # normalized_score = (score - min_score) / (max_score - min_score)
        #return max(0, min(1, normalized_score))  # 确保值在 [0, 1] 范围内
        return score


if __name__ == "__main__":
    # Load attribute definitions and value types from JSON files
    with open("attribute_definitions.json", "r") as f:
        attribute_definitions = json.load(f)

    with open("value_types.json", "r") as f:
        value_types = json.load(f)

    # Instantiate the ObjectSimilarityPlusMethod class
    similarity_calculator = ObjectSimilarityPlusMethod(attribute_definitions, value_types)

    # Folder containing JSON objects
    folder_path = "./testcase/object_level/threat-actor1"  # Replace with the path to your folder

    # Read the primary object from "sample_0.json"
    object1_path = os.path.join(folder_path, "sample_0.json")
    with open(object1_path, "r") as f:
        object1 = json.load(f)

    # Iterate through other JSON files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith("sample") and filename.endswith(".json") and filename != "sample_0.json":
            object2_path = os.path.join(folder_path, filename)
            with open(object2_path, "r") as f:
                object2 = json.load(f)

            # Calculate similarity
            similarity = similarity_calculator.calculate_similarity(object1, object2)

            # Print result
            print(f"Object1: sample_0.json, Object2: {filename}, Similarity: {similarity:.2f}")
