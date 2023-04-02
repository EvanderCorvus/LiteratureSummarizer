import openai
from langchain.llms import OpenAI
import os
import langchain as lc
from langchain.llms import PromptLayerOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from llm_utilities import *
from langchain.callbacks import get_openai_callback

key = os.getenv('OPENAI_API_KEY')
openai.api_key = key

llm_page_extractor = get_llm('text-curie-001', 1)
llm_query_answering = get_llm('text-davinci-003', 0.8)


query = """Welche sind sie die unterschiedlichen Auswahlsregeln für Rotations- und Vibrationsübergänge in Molekülen?""" 

docs = make_query_from_faissindex('PhysicsLiterature', query, 1)
literature = ''

# for doc in docs:
#     relevant_info =  str("source:" + doc.metadata["source"]) + "," + "page:" + str(doc.metadata["page"]) + "content:" + run_page_infoextracter_chain(doc.page_content, query, llm_page_extractor)
#     literature += relevant_info
#     print("Page content: \n\n" + doc.page_content +"\n")
#     print("Relevant content: \n\n" + relevant_info + "\n")
#     #literature += str("source:" + doc.metadata["source"]) + "," + "page:" + str(doc.metadata["page"]) + "content:" + doc.page_content
with get_openai_callback() as cb:
    print(run_query_answering_chain(literature, query, llm_query_answering))
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Successful Requests: {cb.successful_requests}")
    print(f"Total Cost (USD): ${cb.total_cost}")