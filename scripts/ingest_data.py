
# %%
import os
import os.path as osp
import sys
ROOT_DIR = osp.dirname(os.getcwd())
sys.path.append(ROOT_DIR)
# %%
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
from lib.config import  EMBEDDING_MODEL_NAME, TEXTSPLITTER_CHUNK_SIZE,TEXTSPLITTER_OVERLAP

DOCUMENT_PATH='../data/rel18/'
VECTOR_STORE_PATH = '../data/vectorstore/'


def main():
    
    knowledge_files = glob(DOCUMENT_PATH+'*.docx')

    print('Loading documents')
    print(f'chunk_size: {TEXTSPLITTER_CHUNK_SIZE}  overlap: {TEXTSPLITTER_OVERLAP} embedding_model:{EMBEDDING_MODEL_NAME}')
    
    def split_documents(doc_path):
        textsplitter = RecursiveCharacterTextSplitter(chunk_size=TEXTSPLITTER_CHUNK_SIZE, chunk_overlap=TEXTSPLITTER_OVERLAP)
        docs = Docx2txtLoader(doc_path).load_and_split(textsplitter)
        for i in range(len(docs)):
            source = docs[i].metadata['source']
            basename = osp.basename(source)
            if basename.startswith('rel_'):
                year = int(basename.split('_')[1].split('.')[0].strip())
                tag = f'3GPP Release {year}'
                docs[i].metadata['year'] = year
                docs[i].metadata['tag'] = tag
            docs[i].metadata['source'] = basename
            
        return docs

    embeddings = GPT4AllEmbeddings(model_name=EMBEDDING_MODEL_NAME,gpt4all_kwargs ={'allow_download': 'True'},device='gpu')
    
    documents = Parallel(n_jobs=-1,)(delayed(split_documents)(k) for k in tqdm(knowledge_files))

    
    
    async def add_documents(docs):
        try:

            embeddings = GPT4AllEmbeddings(model_name=EMBEDDING_MODEL_NAME,gpt4all_kwargs ={'allow_download': 'True'},device='gpu')
            
            client = chromadb.HttpClient(host='localhost', port=8000)
            await Chroma(client=client,embedding_function=  embeddings).aadd_documents(docs, embedding=embeddings)

        except Exception as e:
            print(f'something went wrong. {e}')

    def run_async_task(docs):
        # Run the async task and wait for its result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(add_documents(docs))
        loop.close()
        return result
        
    print('Adding documents to vectorstore')
    flat_documents = []
    for docs in documents:
        flat_documents += docs

    def chunks(container,size):
        for i in range(0, len(container), size):
            yield container[i:i + size]
            
    Parallel(n_jobs=12)(delayed(run_async_task)(docs) for docs in tqdm(list(chunks(flat_documents,512))))
    

if __name__ == '__main__' :
    main()
    