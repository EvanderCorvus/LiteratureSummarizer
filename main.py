import openai
import os
import langchain as lc
from langchain.llms import OpenAI
# from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

key = os.getenv('OPENAI_API_KEY')
openai.api_key = key
model_name = 'gpt-3.5-turbo'
llm = OpenAI(model_name =   model_name,
             temperature=   1)


fname = 'prompt.txt'
with open(fname,'r') as f:
    template = f.read()

query = 'Tell me in simple words, how the TimeGAN model works.'

faiss_index = FAISS.load_local('MLLiteratur', OpenAIEmbeddings())
docs = faiss_index.similarity_search(query, k=5)
literature = ''
for doc in docs:
    literature += str(doc.metadata["page"]) + ":" + doc.page_content


prompt = lc.PromptTemplate(input_variables=['literature','query'],template=template)

llm_chain = lc.LLMChain(prompt=prompt,llm=llm)

print(llm_chain.run(literature = literature, query=query))