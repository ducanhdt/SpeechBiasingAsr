from vllm import LLM, SamplingParams

inputs = [
    "Here is the largest building in Hanoi",
    "The president of the United States is Trump",
    "The capital of France is Paris",
]


prompt_template = \
"""Extract the address pharse in this sentence,return a list of phrase:

INPUT: I live in New York for 5 years
OUTPUT: ["New York"]

INPUT: I live in Ho Chi Minh City, one of the most crowded city in Vietnam.
OUTPUT: ["Ho Chi Minh City", "Vietnam"]

INPUT:{}
OUTPUT:"""

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# llm = LLM(model="casperhansen/mistral-7b-instruct-v0.1-awq", quantization="awq", dtype="half",**{"max_model_len":2048})
llm = LLM(model="TheBloke/Mistral-7B-v0.1-AWQ", quantization="awq", dtype="half",**{"max_model_len":2048})

inputs = [prompt_template.format(input_) for input_ in inputs]
outputs = llm.generate(inputs, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(prompt)
    print(f"Generated text: {generated_text.strip()}")
print("done")
