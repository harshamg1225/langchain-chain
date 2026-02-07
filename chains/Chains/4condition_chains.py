from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

model2 = ChatOpenAI()


class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative", "neutral"] = Field(
        description="give the sentiment of the feedback"
    )


parser1 = PydanticOutputParser(pydantic_object=Feedback)

template = PromptTemplate(
    template="classify the sentiment of the  feedback into Positive ,Negative and neutral {feedback}.\n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser1.get_format_instructions()},
)


classify_chains = template | model2 | parser1


prompt2 = PromptTemplate(
    template="write an appropriate  response to this postive feedback\n {feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="write an appropriate response to this negative feedback\n {feedback}",
    input_variables=["feedback"],
)


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt2 | model2 | parser),
    (lambda x: x.sentiment == "Negative", prompt3 | model2 | parser),
    RunnableLambda(lambda x: "could not find sentiment"),
)

chain = classify_chains | branch_chain

result = chain.invoke(
    {"feedback": "The food was cold, and the service was incredibly slow"}
)

print(result)
