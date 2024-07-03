from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import gpt4all
import json
import chromadb
from langchain_chroma import Chroma

from lib.deduplicate_retriever import DeduplicateRetriever

def get_retriever(index,
                  reranker_model,
                  embedding_model_name,
                  embedding_model_kwargs,
                  compression_retriever_top_n,
                  vectorstore_host,
                  vectorstore_port,
                  vectorstore_path,
                  vectorstore_k):
    embedding_model_gpu_names   = gpt4all.GPT4All.list_gpus()
    gpu_index = index%len(embedding_model_gpu_names)
    print(f'using gpu: {gpu_index}')
    embeddings = GPT4AllEmbeddings(
        model_name=embedding_model_name,
        gpt4all_kwargs =embedding_model_kwargs,
        device=embedding_model_gpu_names[gpu_index],
    )
    vectorstore_client = get_vectorstore_client(vectorstore_host=vectorstore_host,
                                    vectorstore_port=vectorstore_port,
                                    vectorstore_path=vectorstore_path)
    vectorstore = Chroma(client=vectorstore_client, embedding_function=embeddings)
    vectorstore_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs= {'k': vectorstore_k} 
    )
    #deduplication
    deduplicate_retriever = DeduplicateRetriever(base_retriever=vectorstore_retriever)

    #compression
    rerank_model = HuggingFaceCrossEncoder(model_name=reranker_model, model_kwargs = {'device': f'cuda:{gpu_index}'})

    compressor = CrossEncoderReranker(model=rerank_model, top_n=compression_retriever_top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=deduplicate_retriever
    )

    return compression_retriever


def read_json_file(filename):
    with open(filename,encoding='UTF-8') as json_file :
        json_data = json.load(json_file)
    return json_data


def break_list_into_chunks(container,size):
    for i in range(0, len(container), size):
        yield container[i:i + size]
        
def flatten_list_of_list(container):
    result = []
    for chunk in container:
        result += chunk
    return result 


def get_vectorstore_client(vectorstore_host,vectorstore_port,vectorstore_path):
    if len(vectorstore_host.strip()) > 0 and vectorstore_port > 0:
        print('using http client')
        client = chromadb.HttpClient(host=vectorstore_host, port=vectorstore_port)
    else:
        print('using persistent client')
        client = chromadb.PersistentClient(path=vectorstore_path)
    return client