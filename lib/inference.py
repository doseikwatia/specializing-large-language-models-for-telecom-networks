from joblib import Parallel, delayed
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
import transformers
import torch
import random
from lib.prompt import get_mcq_inference_prompt
from lib.utilities import get_retriever, read_json_file, flatten_list_of_list,break_list_into_chunks
import pandas as pd
import pickle

def _get_question_prompt(rank_id,
                        qstn,        
                        reranker_model,
                        embedding_model_name,
                        embedding_model_kwargs,
                        embedding_dimensionality,
                        compression_retriever_top_n,
                        vectorstore_host,
                        vectorstore_port,
                        vectorstore_path,
                        vectorstore_k
        ): # in tqdm(questions.items()):
    results = []

    retriever = get_retriever(
        rank_id,
        reranker_model,
        embedding_model_name,
        embedding_model_kwargs,
        embedding_dimensionality,
        compression_retriever_top_n,
        vectorstore_host,
        vectorstore_port,
        vectorstore_path,
        vectorstore_k
    )
    backup_retriever = get_retriever(
        rank_id,
        reranker_model,
        embedding_model_name,
        embedding_model_kwargs,
        embedding_dimensionality,
        compression_retriever_top_n,
        vectorstore_host,
        vectorstore_port,
        vectorstore_path,
        int(vectorstore_k/100),
        False
    )
    for qstn_id,qstn_data in tqdm(qstn):
        qstn_id=qstn_id.split(' ')[1].strip()
        qstn_text = qstn_data['question']
        
        #searching through datastore for context
        try:
            docs = retriever.invoke(qstn_text)
        except KeyboardInterrupt:
            break
        except:
            print(f'something went wrong. question: {qstn_text}')
            docs = backup_retriever.invoke(qstn_text)
            
        context =  (' '.join(list(map(lambda d:d.page_content,docs)))).replace('\n', '. ')
        infer_data = get_mcq_inference_prompt(qstn_data, context)
        prompt = infer_data['prompt']
        
        results.append((qstn_id,prompt))
        
        if len(results) % 10 == 1:
            print(f'rank: {rank_id}\n{prompt}')
        
    return results

def _generate_prompts(qst_filename,
                        reranker_model,
                        embedding_model_name,
                        embedding_model_kwargs,
                        embedding_dimensionality,
                        compression_retriever_top_n,
                        vectorstore_host,
                        vectorstore_port,
                        vectorstore_path,
                        vectorstore_k, 
                        sample_size=-1, 
                        n_jobs = 2):
    questions = read_json_file(qst_filename)
    if sample_size < 0 :
        sampled_questions = list(questions.items())
    else:
        sampled_questions = random.sample(list(questions.items()),sample_size)
    
    # sampled_questions.reverse()
    chunk_size = int(len(sampled_questions)/n_jobs) + 1
    print(f'chunk_size: {chunk_size}')
    pprompts = Parallel(n_jobs=n_jobs)(delayed(_get_question_prompt)(rank_id%n_jobs , 
        entry,
        reranker_model,
        embedding_model_name,
        embedding_model_kwargs,
        embedding_dimensionality,
        compression_retriever_top_n,
        vectorstore_host,
        vectorstore_port,
        vectorstore_path,
        vectorstore_k
        ) for rank_id,entry in tqdm(enumerate(break_list_into_chunks(sampled_questions,chunk_size)))) 
    
    prompts = []
    for pprompt in pprompts:
        prompts += pprompt
        
    return prompts
    

def _answer_questions(llm_name,finetunned_model_path,prompts, max_new_tokens=4,return_full_text=False, batch_size = 128):
    solutions = []
    answer_model_name = llm_name
    tokenizer = AutoTokenizer.from_pretrained(answer_model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    # answer_model = AutoModelForCausalLM.from_pretrained(finetunned_model_path,device_map="auto",)
    
    answer_generator = transformers.pipeline(
        "text-generation",
        model=finetunned_model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    num_prompts = len(prompts)
    # ##falcon only
    # for i in tqdm(range(num_prompts)):
    #     prompt = prompts[i][1]
    #     prompt = prompt.replace('### Instructions:','User:')
    #     # prompt = prompt.replace('Context:','<|user|>\nContext:')
    #     prompt = prompt.replace('### Answer:','Assistant:').strip()
    #     prompts[i] = (prompts[i][0],prompt)
    #     if i < 3:
    #         print(prompts[i][1])
    # ##
    
    for i in tqdm(range(0,num_prompts,batch_size)):
        current_prompts=list(map(lambda e:e[1],prompts[i:i+batch_size]))
        current_qstn_ids=list(map(lambda e:e[0],prompts[i:i+batch_size]))
        responses = answer_generator(current_prompts,max_new_tokens=max_new_tokens, return_full_text=return_full_text)
        current_ans_ids =list(map(lambda r:r[0]['generated_text'].split(':')[0][-1:].strip(),responses))
        print(current_ans_ids)
        solutions += list(zip(current_qstn_ids,current_ans_ids))
        
    return solutions    


def _save_solution(filename,solution, task=''):
    df = pd.DataFrame(solution,columns=['Question_ID','Answer_ID'])
    df['Task'] = task
    df.to_csv(filename,index=False,)

def build_inference_prompt(
    run_mode:int,
    training_input_filename:str,
    testing_input_filename:str,
    reranker_model: str,
    embedding_model_name: str,
    embedding_model_kwargs: dict,
    embedding_dimensionality: int,
    compression_retriever_top_n: int,
    vectorstore_host: str,
    vectorstore_port: int,
    vectorstore_path: str,
    vectorstore_k: int,
    training_prompt_bin_filename:str,
    testing_prompt_bin_filename:str,
    n_jobs : int=4,
):
    print(f''' 
run_mode                    = {run_mode}
training_input_filename     = {training_input_filename}
testing_input_filename      = {testing_input_filename}
reranker_model              = {reranker_model}
embedding_model_name        = {embedding_model_name}
embedding_model_kwargs      = {embedding_model_kwargs}
embedding_dimensionality    = {embedding_dimensionality}
compression_retriever_top_n = {compression_retriever_top_n}
vectorstore_host            = {vectorstore_host}
vectorstore_port            = {vectorstore_port}
vectorstore_path            = {vectorstore_path}
vectorstore_k               = {vectorstore_k}
training_prompt_bin_filename= {training_prompt_bin_filename}
testing_prompt_bin_filename = {testing_prompt_bin_filename}
n_jobs                      = {n_jobs}
          ''')
    training_io = (training_input_filename,training_prompt_bin_filename)
    testing_io  = (testing_input_filename,testing_prompt_bin_filename)
    
    if run_mode == 2:
        io_filenames = [training_io, testing_io ]
    elif run_mode == 1:
        io_filenames = [testing_io]
    else:
        io_filenames = [training_io]
    
    for i_filename, o_filename in io_filenames:
        print(f'processing {i_filename} - {o_filename}')
        prompts = _generate_prompts(
            i_filename,
            reranker_model,
            embedding_model_name,
            embedding_model_kwargs,
            embedding_dimensionality,
            compression_retriever_top_n,
            vectorstore_host,
            vectorstore_port,
            vectorstore_path,
            vectorstore_k, 
            sample_size=-1, 
            n_jobs=n_jobs)
        with open(o_filename,'b+w') as f:
            pickle.dump(prompts,f)
    
    print('build prompt done')
    
    
    
def test_model(
    task:str,                      
    run_mode:int,
    llm_name:str,
    final_model_output_dir:str,
    training_output_filename:str,
    testing_output_filename:str,
    training_prompt_bin_filename:str,
    testing_prompt_bin_filename:str,
):    
    print(f''' 
task                            = {task}  
run_mode                        = {run_mode}
llm_name                        = {llm_name}
final_model_output_dir          = {final_model_output_dir}
training_output_filename        = {training_output_filename}
testing_output_filename         = {testing_output_filename}
training_prompt_bin_filename    = {training_prompt_bin_filename}
testing_prompt_bin_filename     = {testing_prompt_bin_filename}
          ''')
    training_io = (training_prompt_bin_filename,training_output_filename)
    testing_io  = (testing_prompt_bin_filename,testing_output_filename)
    
    if run_mode == 2:
        io_filenames = [training_io, testing_io]
    elif run_mode == 1:
        io_filenames = [testing_io]
    else:
        io_filenames = [training_io]
    
    for i_filename,o_filename in io_filenames:
        print(f'processing {i_filename} - {o_filename}')
        
        with open(i_filename,'b+r') as f:
            prompts = pickle.load(f)

        train_soln = _answer_questions(
            llm_name=llm_name,
            finetunned_model_path=final_model_output_dir,
            prompts=prompts,
            batch_size = 512)

        _save_solution(o_filename,train_soln,task)
    
    print('test model is done')


def calculate_accuracy(
    training_output_filename:str,
    training_results:str,
):
    pred = pd.read_csv(training_output_filename)
    act = pd.read_csv(training_results)
    act = act[act['Question_ID'].isin(pred['Question_ID'])]

    pred=pred.sort_values(by="Question_ID").reset_index(drop=True)
    act=act.sort_values(by="Question_ID").reset_index(drop=True)
    
    pred['Answer_ID']=pred['Answer_ID'].astype(str)
    act['Answer_ID']=act['Answer_ID'].astype(str)
    
    accuracy= (pred['Answer_ID'] == act['Answer_ID']).mean()
    
    print(f'Accuracy: {accuracy}')
