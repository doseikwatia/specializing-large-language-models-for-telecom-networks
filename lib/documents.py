from chromadb import Documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from lib.config import EMBEDDING_MODEL_NAME, TEXTSPLITTER_CHUNK_SIZE, TEXTSPLITTER_OVERLAP, GPU_NAMES
from langchain_community.embeddings import GPT4AllEmbeddings
import os.path as osp
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import List

EMBEDDING_KWARGS = {'allow_download': 'True'}

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


def load_documents(dirname:str)->List[Documents]:
    knowledge_files = glob(dirname+'*.docx')
    # gpu_device = GPU_NAMES[index%len(GPU_NAMES)]
    # GPT4AllEmbeddings(model_name=EMBEDDING_MODEL_NAME,gpt4all_kwargs=EMBEDDING_KWARGS,device=gpu_device)
    
    documents = Parallel(n_jobs=-1,)(delayed(split_documents)(k) for k in tqdm(knowledge_files))
    return documents