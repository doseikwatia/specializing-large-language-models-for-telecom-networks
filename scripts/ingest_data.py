
from glob import glob
from langchain_chroma import Chroma
from langchain_community.vectorstores import SQLiteVSS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from tqdm import tqdm
from joblib import Parallel, delayed
import chromadb
import asyncio

DOCUMENT_PATH='../data/rel18/'
VECTOR_STORE_PATH = '../data/vectorstore/'
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50


def main():
    
    knowledge_files = glob(DOCUMENT_PATH+'*.docx')

    textsplitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    print('Loading documents')
    print(f'chunk_size: {CHUNK_SIZE}  overlap: {CHUNK_OVERLAP}')
    
    documents = Parallel(n_jobs=-1,)(delayed(lambda k: Docx2txtLoader(k).load_and_split(textsplitter))(k) for k in tqdm(knowledge_files))

    
    
    async def add_documents(docs):
        try:
            embeddings = GPT4AllEmbeddings(
            model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
            device='gpu',
            gpt4all_kwargs={'allow_download': 'True'}
            )
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
            
    Parallel(n_jobs=16)(
        delayed(run_async_task)(docs) for docs in tqdm(list(chunks(flat_documents,512))))
    

if __name__ == '__main__' :
    main()
    

    # Chroma.from_documents(
    #     documents=docs,
    #     embedding=  embeddings,
    #     persist_directory=VECTOR_STORE_PATH+"chromadb_store_256"
    # )
    # store = SQLiteVSS(
    #         connection=sql_connection,
    #         embedding=embeddings,
    #         table="documents"
    #     )
    # store.add_documents(docs)

    # sql_connection  = SQLiteVSS.create_connection(db_file=VECTOR_STORE_PATH+"chromadb_store_256.db")


            # Chroma(persist_directory=VECTOR_STORE_PATH+"chromadb_store",embedding_function=  embeddings).add_documents(docs, embedding=embeddings)

            # sql_connection  = SQLiteVSS.create_connection(db_file=VECTOR_STORE_PATH+f"chromadb_store.db")
            # store = SQLiteVSS(
            #         connection=sql_connection,
            #         embedding=embeddings,
            #         table="documents"
            # )