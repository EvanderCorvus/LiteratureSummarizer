import openai
import os
import langchain as lc
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader

key = os.getenv('OPENAI_API_KEY')
openai.api_key = key
model_name = 'gpt-3.5-turbo'
llm = OpenAI(model_name =   model_name,
             temperature=   1)
loader = UnstructuredPDFLoader('literatur/Nolting, Elektrodynamik',
                               mode = 'elements')
literature = loader.load()

fname = 'prompt.txt'
with open(fname,'r') as f:
    template = f.read()

query = 'what are the maxwell equations in integral form?'

prompt = lc.PromptTemplate(input_variables=['literature','query'],template=template)

llm_chain = lc.LLMChain(prompt=prompt,llm=llm)

print(llm_chain.run(literature = literature, query=query))