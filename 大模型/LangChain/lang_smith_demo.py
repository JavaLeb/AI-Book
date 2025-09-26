from langsmith import Client

# 'inputs' will come from your dataset.
def dummy_target(inputs: dict) -> dict:
    return {"foo": 1, "bar": "two"}

# 'inputs' will come from your dataset.
# 'outputs' will come from your target function.
def evaluator_one(inputs: dict, outputs: dict) -> bool:
    return outputs["foo"] == 2

def evaluator_two(inputs: dict, outputs: dict) -> bool:
    return len(outputs["bar"]) < 3

client = Client()
results = client.evaluate(
    dummy_target,  # <-- target function
    data="your-dataset-name",
    evaluators=[evaluator_one, evaluator_two],
    ...
)