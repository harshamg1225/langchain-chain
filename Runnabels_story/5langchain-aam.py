import random

## compeonets for llm


class NakliLLM:
    def __init__(self):
        print("LLm created")

    def predict(self, prompt):

        response_list = [
            "Delhi is the capital of india",
            "ipl is a cricket league",
            "Ai stands for artifical intelligence",
        ]

        return {"response": random.choice(response_list)}


llm = NakliLLM()

# print(llm.predict("what is capital of india"))


# componets for prompt


class NakliPromptTemplate:
    def __init__(self, template, input_variable):

        self.template = template
        self.input_variable = input_variable

    def format(self, input_dict):

        return self.template.format(**input_dict)


# template = NakliPromptTemplate(template="what is {topic}", input_variable="topic")

# prompt = template.format({"topic": "AI"})
# print(prompt)

template1 = NakliPromptTemplate(template="what is {topic}", input_variable="topic")

prompt1 = template1.format({"topic": "AI"})

llm1 = NakliLLM()

result = llm1.predict(prompt=prompt1)
print(result)
