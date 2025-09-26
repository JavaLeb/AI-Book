from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
""",
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
""",
    },
]
prompt_value = prompt.invoke(examples[0])
print(prompt_value.to_string())

print('---------------------------------FewShotPromptTemplate--------------------------------------')
from langchain_core.prompts import FewShotPromptTemplate

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

print(
    few_shot_prompt.invoke({"input": "Who was the father of Mary Ball Washington?"}).to_string()
)
print('-----------------------------------example_selector------------------------------------------')
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_ollama import OllamaEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, OllamaEmbeddings(model='nomic-embed-text'), Chroma, k=1, )

# Select the most similar example to the input.
question = "Who was the father of Mary Ball Washington?"
selected_examples = example_selector.select_examples({"question": question})
print(f"Examples most similar to the input: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

print(
    prompt.invoke({"input": "Who was the father of Mary Ball Washington?"}).to_string()
)
