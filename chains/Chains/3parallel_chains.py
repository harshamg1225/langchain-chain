from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatOpenAI()

parser = StrOutputParser()

template1 = PromptTemplate(
    template="generates notes on this {text}", input_variables=["text"]
)

template2 = PromptTemplate(
    template="generate 5 question from the text {text}", input_variables=["text"]
)


chain1 = template1 | model1 | parser
chain2 = template2 | model2 | parser

output_parallel = RunnableParallel({"notes": chain1, "question": chain2})

# template for combing both paragraph, question
template3 = PromptTemplate(
    template=""""Combine the following content EXACTLY into two sections. 

### Notes
{notes}

### Questions
{question}

Do not summarize. Keep both sections clearly separated.""",
    input_variables=["notes", "question"],
)


merge_chain = template3 | model2 | parser

chain = output_parallel | merge_chain

text = """Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates"""

result = chain.invoke({"text": text})

print(result)
