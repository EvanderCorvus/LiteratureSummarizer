import openai
import os
import langchain as lc
from langchain.llms import PromptLayerOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

key = os.getenv('OPENAI_API_KEY')
openai.api_key = key
model_name = 'gpt-4-0314'
llm = OpenAI(model_name =   model_name,
             temperature=   1)


fname = 'prompt.txt'
with open(fname,'r') as f:
    template = f.read()

query = """ Explain the holographic principle, to someone who has only machine learning, neural network background and knows only a little bit about quantum theory.""" 

faiss_index = FAISS.load_local('MLLiteratur', OpenAIEmbeddings())
docs = faiss_index.similarity_search(query, k=20)
literature = ''

for doc in docs:
    literature += str("source:" + doc.metadata["source"]) + "," + "page:" + str(doc.metadata["page"]) + "content:" + doc.page_content

prompt = lc.PromptTemplate(input_variables=['literature','query'],template=template)

llm_chain = lc.LLMChain(prompt=prompt,llm=llm)

print(llm_chain.run(literature = literature, query=query))