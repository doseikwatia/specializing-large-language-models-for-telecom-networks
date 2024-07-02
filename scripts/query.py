# %%
# %%
import os
import os.path as osp
import sys
ROOT_DIR = osp.dirname(os.getcwd())
sys.path.append(ROOT_DIR)

# %% [markdown]
# # Set up models, vectorstore and retriever

# %%
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from tqdm import tqdm
from lib.config import LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, VECTOR_STORE_NAME,COMPRESSION_RETRIEVER_TOP_N,VECTOR_RETRIEVER_K,RERANKER_MODEL_NAME,GPU_NAMES
from lib.deduplicate_retriever import DeduplicateRetriever

# %%
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
import transformers
import torch
# %%
DOCUMENT_PATH='../data/rel18/'
VECTOR_STORE_PATH = '../data/vectorstore/'
EMBEDDING_KWARGS = {'allow_download': 'True'}



# %%
def get_retriever(index):
    gpu_index = index%len(GPU_NAMES)
    print(f'using gpu: {gpu_index}')
    embeddings = GPT4AllEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        gpt4all_kwargs =EMBEDDING_KWARGS,
        device=GPU_NAMES[gpu_index],
    )
    
    vectorstore = Chroma(persist_directory=VECTOR_STORE_PATH+VECTOR_STORE_NAME, embedding_function=embeddings)
    vstore_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs= {'k': VECTOR_RETRIEVER_K} 
    )
    vstore_retriever = DeduplicateRetriever(base_retriever=vstore_retriever)

    #compression
    rerank_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME, model_kwargs = {'device': f'cuda:{gpu_index}'})

    compressor = CrossEncoderReranker(model=rerank_model, top_n=COMPRESSION_RETRIEVER_TOP_N)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=vstore_retriever
    )

    return compression_retriever


# %%
print(VECTOR_STORE_NAME)
print(VECTOR_RETRIEVER_K)
print(COMPRESSION_RETRIEVER_TOP_N)


# %%
from lib.prompt import get_mcq_inference_prompt
import json
import pandas as pd
from joblib import Parallel, delayed
import pickle

# %%
import random
len({'j':6})

# %%
def chunks(container,size):
    for i in range(0, len(container), size):
        yield container[i:i + size]
        
def get_question_prompt(rank_id,qstn): # in tqdm(questions.items()):
    results = []
    retriever = get_retriever(rank_id)
    for qstn_id,qstn_data in tqdm(qstn):
        qstn_id=qstn_id.split(' ')[1].strip()
        qstn_text = qstn_data['question']
        #searching through datastore for context
        docs = retriever.invoke(qstn_text)
        context =  (' '.join(list(map(lambda d:d.page_content,docs)))).replace('\n', '. ')
        infer_data = get_mcq_inference_prompt(qstn_data, context)
        prompt = infer_data['prompt']
        
        results.append((qstn_id,prompt))
        
    del retriever
    
    return results

def generate_prompts(qst_filename, sample_size=-1):
    n_jobs = 2
    with open(qst_filename) as file:
        questions = json.load(file)
    
    if sample_size < 0 :
        sampled_questions = list(questions.items())
    else:
        sampled_questions = random.sample(list(questions.items()),sample_size)
    
    sampled_questions.reverse()
    chunk_size = int(len(sampled_questions)/n_jobs) + 1
    print(f'chunk_size: {chunk_size}')
    pprompts = Parallel(n_jobs=n_jobs)(delayed(get_question_prompt)(rank_id%n_jobs , entry) for rank_id,entry in tqdm(enumerate(chunks(sampled_questions,chunk_size)))) 
    
    prompts = []
    for pprompt in pprompts:
        prompts += pprompt
        
    return prompts
    
    
def answer_questions(prompts, max_new_tokens=4,return_full_text=False, batch_size = 128):
    solutions = []
    answer_model_name = LLM_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(answer_model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    answer_model = AutoModelForCausalLM.from_pretrained('../bin/pretrained_512_32/',device_map="auto",)
    answer_generator = transformers.pipeline(
        "text-generation",
        model=answer_model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    num_prompts = len(prompts)
    for i in tqdm(range(0,num_prompts,batch_size)):
        current_prompts=list(map(lambda e:e[1],prompts[i:i+batch_size]))
        current_qstn_ids=list(map(lambda e:e[0],prompts[i:i+batch_size]))
        responses = answer_generator(current_prompts,max_new_tokens=max_new_tokens, return_full_text=return_full_text)
        current_ans_ids =list(map(lambda r:r[0]['generated_text'].split(':')[0][-1:].strip(),responses))
        solutions += list(zip(current_qstn_ids,current_ans_ids))
        
    return solutions

# %%
def save_solution(filename,solution, task=''):
    df = pd.DataFrame(solution,columns=['Question_ID','Answer_ID'])
    df['Task'] = task
    df.to_csv(filename,index=False,)

# %%
# prompts = generate_prompts('../data/Question_Submission.txt', sample_size=-1)
# with open('prompts.bin','b+w') as f:
#     pickle.dump(prompts,f)

with open('prompts.bin','b+r') as f:
    prompts = pickle.load(f)

print(prompts[1])
# %%
# with open('prompts.bin','b+r') as f:
#     prompts = pickle.load(f)
train_soln = answer_questions(prompts,batch_size = 128)


# %%
save_solution('testing_result.csv',train_soln,'Phi-2')
