# Function to check if the type attribute matches and assign the corresponding score.
def match_object_type(obj1, obj2):
    if obj1.get("type") == obj2.get("type"):
        return 0.5
    else:
        return 0.0


# Function to match attribute types based on required and optional attributes.
def match_attribute_type(obj1, obj2, attribute_weights):
    # Retrieve object type
    obj_type = obj1.get("type")

    # Check if object type exists in the dictionary
    if obj_type not in attribute_weights:
        return 0.0

    weights = attribute_weights[obj_type]
    required_attrs = weights["required"]
    optional_weight = weights["optional"]

    score = 0.0

    # Check required attributes
    for attr, weight in required_attrs.items():
        if obj1.get(attr) == obj2.get(attr):
            score += weight

    # Check optional attributes, excluding 'id' and 'spec_version'
    excluded_attrs = {"id", "spec_version", "type"}
    optional_attrs = set(obj1.keys()).union(obj2.keys()) - set(required_attrs.keys()) - excluded_attrs

    if optional_attrs:
        optional_score_per_attr = optional_weight / len(optional_attrs)
        for attr in optional_attrs:
            if obj1.get(attr) == obj2.get(attr):
                score += optional_score_per_attr

    return score


# Combine the two scores into a final similarity score
def calculate_similarity_score(obj1, obj2, attribute_weights):
    type_score = match_object_type(obj1, obj2)
    attribute_score = match_attribute_type(obj1, obj2, attribute_weights)
    return type_score + attribute_score
