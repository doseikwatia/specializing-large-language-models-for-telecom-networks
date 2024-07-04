
from glob import glob
import os.path as osp
from typing import List
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from joblib import Parallel, delayed
import gpt4all
from lib.utilities import get_vectorstore_client,break_list_into_chunks,flatten_list_of_list

def split_documents(doc_path, chunk_size, overlap):
    '''
    Splits document using the RecursiveCharacterTextSplitter.
    '''
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = Docx2txtLoader(doc_path).load_and_split(textsplitter)
    for i, doc in enumerate(docs):
        source = doc.metadata['source']
        basename = osp.basename(source)
        if basename.startswith('rel_'):
            year = int(basename.split('_')[1].split('.')[0].strip())
            tag = f'3GPP Release {year}'
            docs[i].metadata['year'] = year
            docs[i].metadata['tag'] = tag
        docs[i].metadata['source'] = basename
        
    return docs


def add_documents(index,docs,vectorstore_path:str,vectorstore_host:str,vectorstore_port:int,embedding_model_gpu_names:List[str],embedding_model_name:str,embedding_model_kwargs:dict,subchunk:int=64):
    '''
    Adds document to vector store
    '''
    try:
        gpu_index = index % len(embedding_model_gpu_names)
        gpu_device = embedding_model_gpu_names[gpu_index]
        embeddings = GPT4AllEmbeddings(model_name=embedding_model_name,gpt4all_kwargs =embedding_model_kwargs,device=gpu_device)
        
        client = get_vectorstore_client(vectorstore_host=vectorstore_host,
                                        vectorstore_port=vectorstore_port,
                                        vectorstore_path=vectorstore_path)
        db = Chroma(client=client,embedding_function= embeddings)
        
        groups = list(break_list_into_chunks(docs,subchunk))
        for group_id in tqdm(range(len(groups))):
            db.add_documents(groups[group_id])
            
    except Exception as e:
        print(f'something went wrong. {e}')
        
    print(f'Subprocess {index} has completed')


def load_documents(embedding_model_name:str,
                   embedding_model_kwargs:dict,
                   vectorstore_path:str,
                   vectorstore_host:str,
                   vectorstore_port:int,
                   textsplitter_chunk_size:int,
                   textsplitter_overlap:int,
                   documents_path: str,
                   documents_extentions: List[str],
                   n_jobs:int = 24,
                   index_chunk:int = 64):

    embedding_model_gpu_names   = gpt4all.GPT4All.list_gpus()

    print(f'''Loading data into vectorstore with the following details
Embedding Model:    {embedding_model_name}
Gpu_Names:          {embedding_model_gpu_names}
Vectorstore Path:   {vectorstore_path}
Document Path:      {documents_path}
Document Extensions:{documents_extentions}
''')
    print('Discovering files to be loaded')
    knowledge_files = []
    documents_path = documents_path.strip()
    documents_dir = documents_path if documents_path.endswith('/') else documents_path+'/'
    for extension in documents_extentions:
        knowledge_files  += glob(documents_dir+extension)
        
    print(f'Discovered files, {knowledge_files}')
    
    print('Breaking documents into chunks')
    documents = Parallel(n_jobs=-1,)(delayed(split_documents)(k,textsplitter_chunk_size,textsplitter_overlap) for k in tqdm(knowledge_files))
        
    
    #flatten documents 
    flat_documents = flatten_list_of_list(documents)
            
    job_size =  int(len(flat_documents)/n_jobs)+1
    
    print(f'Saving documents into vectorstore. {n_jobs} jobs are running with job size, {job_size}')
    
    
    Parallel(n_jobs=n_jobs)(delayed(add_documents)(rank_id % n_jobs,docs,vectorstore_path,vectorstore_host,vectorstore_port,embedding_model_gpu_names,embedding_model_name,embedding_model_kwargs,index_chunk) for rank_id,docs in tqdm(enumerate(list(break_list_into_chunks(flat_documents,job_size)))))
    
