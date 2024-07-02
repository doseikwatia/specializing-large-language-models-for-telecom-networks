
#%%
import os
import os.path as osp
import sys

ROOT_DIR = osp.dirname(os.getcwd())
sys.path.append(ROOT_DIR)

from lib.documents import load_documents
from glob import glob
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import GPT4AllEmbeddings


from tqdm import tqdm
from joblib import Parallel, delayed
import chromadb
import asyncio
import os.path as osp
from lib.config import  EMBEDDING_MODEL_NAME, VECTOR_STORE_NAME,COMPRESSION_RETRIEVER_TOP_N,VECTOR_RETRIEVER_K,RERANKER_MODEL_NAME
from lib.normic_wrapper import NomicEmbedding



DOCUMENT_PATH='../data/rel18/'
VECTOR_STORE_PATH = '../data/vectorstore/'
EMBEDDING_KWARGS = {'allow_download': 'True'}

embeddings = GPT4AllEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    gpt4all_kwargs =EMBEDDING_KWARGS,
    device='gpu',
)

vectorstore = Chroma(persist_directory=VECTOR_STORE_PATH+VECTOR_STORE_NAME, embedding_function=embeddings)
#%%
documents = vectorstore.get(limit=10000000,offset=0)
#%%
print(documents.keys())
print(len(documents['ids']))
print(documents['documents'][0])
# %%
num_docs = len(documents['documents'])
unique_documents = {}
duplicates = []

for i in tqdm(range(num_docs)):
    content = documents['documents'][i].strip()
    doc_id = documents['ids'][i]
    if content in unique_documents:
        duplicates.append(doc_id)
    else:
        unique_documents[content] = doc_id

# %%
print(f'number of duplicates found: {len(duplicates)}')
# %%
# Step 3: Delete duplicates
for duplicate_id in tqdm(duplicates):
    vectorstore.delete((duplicate_id))
# %%
