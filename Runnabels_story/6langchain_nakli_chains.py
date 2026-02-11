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


# prompttemplate


class NakliPromptTemplate:
    def __init__(self, template, input_variable):

        self.template = template
        self.input_variable = input_variable

    def format(self, input_dict):

        return self.template.format(**input_dict)


# instiate the variable

llm = NakliLLM()
template = NakliPromptTemplate(template="what is {topic}", input_variable="topic")

# nakli chains


class NakliLLmchain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):

        final_prompt = self.prompt.format(input_dict)

        result = llm.predict(final_prompt)

        return result


chains = NakliLLmchain(llm=llm, prompt=template)

input_dict = {"topic": "AI"}
result = chains.run(input_dict=input_dict)
print(result)
