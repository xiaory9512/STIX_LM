import json
import os
import csv
from object_similarity import ObjectSimilarity


def main():
    # Load attribute definitions and value types
    with open("attribute_definitions.json", "r") as f:
        attribute_definitions = json.load(f)

    with open("value_types.json", "r") as f:
        value_types = json.load(f)

    # Instantiate similarity calculator
    similarity_calculator = ObjectSimilarity(attribute_definitions, value_types)

    # Prepare for comparison
    folder_path = "./testcase/object_level/threat-actor1/"
    base_file = os.path.join(folder_path, "sample_0.json")
    with open(base_file, "r") as f:
        base_object = json.load(f)

    output_file = "similarity_results_threatactor1.csv"

    # Write results to CSV
    with open(output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ["File", "Basic COS", "Basic BLEU", "Advanced COS", "Advanced BLEU"]
        csvwriter.writerow(header)

        for i in range(1, 10):
            comparison_file = os.path.join(folder_path, f"sample_{i}.json")
            with open(comparison_file, "r") as f:
                comparison_object = json.load(f)

            basic_cos = similarity_calculator.basic_similarity(base_object, comparison_object, method="COS")
            basic_bleu = similarity_calculator.basic_similarity(base_object, comparison_object, method="BLEU")
            advanced_cos = similarity_calculator.advanced_similarity(
                base_object, comparison_object, base_object["type"], method="COS"
            )
            advanced_bleu = similarity_calculator.advanced_similarity(
                base_object, comparison_object, base_object["type"], method="BLEU"
            )

            csvwriter.writerow([f"{i}.json", basic_cos, basic_bleu, advanced_cos, advanced_bleu])

    print(f"Similarity results written to {output_file}")


if __name__ == "__main__":
    main()
