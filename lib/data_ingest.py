
from lib.documents import load_documents
from glob import glob
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

from tqdm import tqdm
from joblib import Parallel, delayed
import chromadb
import asyncio
import os.path as osp
from lib.config import  EMBEDDING_MODEL_NAME, TEXTSPLITTER_CHUNK_SIZE,TEXTSPLITTER_OVERLAP, GPU_NAMES

DOCUMENT_PATH='../data/rel18/'
VECTOR_STORE_PATH = '../data/vectorstore/'
EMBEDDING_KWARGS = {'allow_download': 'True'}

def main():
    print('Loading documents')
    print(f'chunk_size: {TEXTSPLITTER_CHUNK_SIZE}  overlap: {TEXTSPLITTER_OVERLAP} embedding_model:{EMBEDDING_MODEL_NAME}')
    
    documents = load_documents(DOCUMENT_PATH)
    
    
    async def add_documents(index,docs):
        try:
            gpu_index = index % len(GPU_NAMES)
            gpu_device = GPU_NAMES[gpu_index]
            embeddings = GPT4AllEmbeddings(model_name=EMBEDDING_MODEL_NAME,gpt4all_kwargs =EMBEDDING_KWARGS,device=gpu_device)
            client = chromadb.HttpClient(host='localhost', port=8000)
            await Chroma(client=client,embedding_function=  embeddings).aadd_documents(docs)

        except Exception as e:
            print(f'something went wrong. {e}')

    def run_async_task(rank_id,docs):
        # Run the async task and wait for its result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(add_documents(rank_id,docs))
        loop.close()
        return result
        
    print('Adding documents to vectorstore')
    flat_documents = []
    for docs in documents:
        flat_documents += docs

    def chunks(container,size):
        for i in range(0, len(container), size):
            yield container[i:i + size]
    n_jobs = 4
    Parallel(n_jobs=n_jobs)(delayed(run_async_task)(rank_id % n_jobs,docs) for rank_id,docs in tqdm(enumerate(list(chunks(flat_documents,512)))))
    

if __name__ == '__main__' :
    main()
    