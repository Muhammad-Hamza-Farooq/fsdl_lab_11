"""Task 23 - model/API version compatibility checks."""

SCHEMA_V1 = {"input": "list[float]", "output": {"prediction": "float", "version": "v1"}}
SCHEMA_V2 = {"input": "list[float]", "output": {"prediction": "float", "confidence": "float", "version": "v2"}}


def compare_schemas(old, new):
    removed = set(old["output"]) - set(new["output"])
    added = set(new["output"]) - set(old["output"])
    return {"removed": list(removed), "added": list(added)}


def validate_contract(schema, response):
    required = schema["output"]
    return all(key in response for key in required)


def predict_v1(data):
    return {"prediction": sum(data), "version": "v1"}


def predict_v2(data):
    total = sum(data)
    return {"prediction": total, "confidence": 0.95, "version": "v2"}


def compatibility_test(client_response, schema):
    return validate_contract(schema, client_response)


def main():
    print("=== Task 23: Model Version Compatibility ===\n")

    diff = compare_schemas(SCHEMA_V1, SCHEMA_V2)
    print("Schema diff:", diff)

    v1_response = predict_v1([1, 2, 3])
    v2_response = predict_v2([1, 2, 3])

    print("V1 contract ok:", compatibility_test(v1_response, SCHEMA_V1))
    print("V2 contract ok:", compatibility_test(v2_response, SCHEMA_V2))
    print("V1 client vs V2 response:", compatibility_test(v2_response, SCHEMA_V1))


if __name__ == "__main__":
    main()
