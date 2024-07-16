import chromadb.config
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker, LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever,MergerRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import gpt4all
import json
import chromadb
from langchain_chroma import Chroma

from lib.constants import ALL_MINILM_L6_V2, NOMIC_EMBED_TEXT_V1, NOMIC_EMBED_TEXT_V1_5, MULTI_QA_MINILM_L6_DOT_V1
from lib.deduplicate_retriever import DeduplicateRetriever
from lib.final_doc_compressor import FinalDocumentCompressor
from lib.normic_wrapper import NomicEmbedding
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def get_retriever(index,
                  reranker_model,
                  embedding_model_name,
                  embedding_model_kwargs,
                  embedding_dimensionality,
                  compression_retriever_top_n,
                  vectorstore_host,
                  vectorstore_port,
                  vectorstore_path,
                  vectorstore_k,
                  use_gpu=True):
    '''
    returns the retriever used in our solution
    '''
    embedding_model_gpu_names   = gpt4all.GPT4All.list_gpus()
    gpu_index = index%len(embedding_model_gpu_names)
    gpu_device = embedding_model_gpu_names[gpu_index] if use_gpu else 'cpu'
    cuda_device = f'cuda:{gpu_index}' if use_gpu else 'cpu'
    print(f'rank_id: {index} is using gpu: {gpu_index}')

    if embedding_model_name == ALL_MINILM_L6_V2:
        embeddings = GPT4AllEmbeddings(model_name=embedding_model_name,gpt4all_kwargs =embedding_model_kwargs,device=gpu_device)
    elif embedding_model_name == NOMIC_EMBED_TEXT_V1 or embedding_model_name == NOMIC_EMBED_TEXT_V1_5:
        embeddings = NomicEmbedding(model_name=embedding_model_name,device=gpu_device, dimensionality=embedding_dimensionality)
    elif embedding_model_name == MULTI_QA_MINILM_L6_DOT_V1:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name,model_kwargs = {'device':cuda_device })
    else:
        raise ValueError('invalid embedding name specified')
    
    vectorstore_client = get_vectorstore_client(vectorstore_host=vectorstore_host,
                                    vectorstore_port=vectorstore_port,
                                    vectorstore_path=vectorstore_path)
    vectorstore = Chroma(client=vectorstore_client, embedding_function=embeddings)
    
    vectorstore_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': vectorstore_k, }
    )
    
    # vectorstore_retriever_mmr = vectorstore.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={'k': vectorstore_k, }
    # )
    # vectorstore_retriever = MergerRetriever(retrievers=[vectorstore_retriever_sim, vectorstore_retriever_mmr])
    
    #deduplication
    deduplicate_retriever = DeduplicateRetriever(base_retriever=vectorstore_retriever)

    #compression
    rerank_model = HuggingFaceCrossEncoder(model_name=reranker_model, model_kwargs = {'device': f'cuda:{gpu_index}'})

    compressor = CrossEncoderReranker(model=rerank_model, top_n=compression_retriever_top_n)

    
    # splitter = CharacterTextSplitter(chunk_size=512,chunk_overlap=0, separator=". ")
    # relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.4)
    # redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    # splitter,redundant_filter,relevant_filter,
    long_ctx_reorder= LongContextReorder()
    final_compress = FinalDocumentCompressor()
    pipeline = DocumentCompressorPipeline(transformers=[compressor,long_ctx_reorder,final_compress])
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=deduplicate_retriever
    )

    return compression_retriever


def read_json_file(filename):
    '''
    reads json file
    '''
    with open(filename,encoding='UTF-8') as json_file :
        json_data = json.load(json_file)
    return json_data


def break_list_into_chunks(container,size):
    '''
    Breaks list up into chunks of the size provided. 
    The last item could have a smaller
    '''
    for i in range(0, len(container), size):
        yield container[i:i + size]
        
def flatten_list_of_list(container):
    '''
    flattens a list of lists into a single list eg.
    input [[a,b],[c,d]]
    output [a,b,c,d]
    '''
    result = []
    for chunk in container:
        result += chunk
    return result 


def get_vectorstore_client(vectorstore_host,vectorstore_port,vectorstore_path):
    # settings = chromadb.config.Settings(    hnsw_params={
    #     'M': 32,             # Maximum connections for each node
    #     'ef_construction': 200,  # Effective search for construction
    #     'ef_search': 50      # Effective search during querying
    # })

    if len(vectorstore_host.strip()) > 0 and vectorstore_port > 0:
        print('using http client')
        client = chromadb.HttpClient(host=vectorstore_host, port=vectorstore_port)
    else:
        print('using persistent client')
        client = chromadb.PersistentClient(path=vectorstore_path)
    return client
