import random
from abc import ABC, abstractmethod


class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass


## compeonets for llm
class NakliLLM(Runnable):
    def __init__(self):
        print("LLm created")

    def invoke(self, prompt):
        response_list = [
            "Delhi is the capital of india",
            "ipl is a cricket league",
            "Ai stands for artifical intelligence",
        ]

        return {"response": random.choice(response_list)}

    def predict(self, prompt):

        response_list = [
            "Delhi is the capital of india",
            "ipl is a cricket league",
            "Ai stands for artifical intelligence",
        ]

        return {"response": random.choice(response_list)}


# prompttemplate
class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variable):

        self.template = template
        self.input_variable = input_variable

    # invoke method
    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):

        return self.template.format(**input_dict)


class NakliStringOuputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):

        return input_data["response"]


class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):

        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data


# instiate the variable

llm = NakliLLM()
template = NakliPromptTemplate(template="what is {topic}", input_variable="topic")
parser = NakliStringOuputParser()


chain = RunnableConnector([template, llm, parser])

result = chain.invoke({"topic": "AI"})
print(result)
