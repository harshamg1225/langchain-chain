from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableMap
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="generate complete story about topic {topic}", input_variables=["topic"]
)
template2 = PromptTemplate(
    template="generate a linkedin post about the topic {topic}",
    input_variables=["topic"],
)


parser = StrOutputParser()

runnable = RunnableMap(
    {
        "story": RunnableSequence(template1, model, parser),
        "linkedin": RunnableSequence(template2, model, parser),
    }
)

result = runnable.invoke({"topic": "Ml"})

print(result["linkedin"])
