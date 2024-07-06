from transformers import AutoTokenizer
import numpy as np
from datasets import Dataset,DatasetDict
from tqdm import tqdm

from lib.prompt import get_mcq_training_prompt, get_qa_training_prompt
from joblib import Parallel, delayed
import pickle
from lib.utilities import break_list_into_chunks, flatten_list_of_list, get_retriever, read_json_file

from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from peft import LoftQConfig, LoraConfig, get_peft_model
from lib.prompt import train_response_template, train_instruction_template
from trl import  SFTTrainer, DataCollatorForCompletionOnlyLM 

def _get_prompt(index,
               qstn_datas,
               reranker_model,
               embedding_model_name,
               embedding_model_kwargs,
               compression_retriever_top_n,
               vectorstore_path,
               vectorstore_host,
               vectorstore_port,
               vectorstore_k):
    retriever = get_retriever(
        index=index,
        reranker_model=reranker_model,
        embedding_model_name=embedding_model_name,
        embedding_model_kwargs=embedding_model_kwargs,
        compression_retriever_top_n = compression_retriever_top_n,
        vectorstore_path=vectorstore_path,
        vectorstore_host=vectorstore_host,
        vectorstore_port=vectorstore_port,
        vectorstore_k=vectorstore_k
        )
    prompts = []
    for qstn_data in tqdm(qstn_datas):
        qstn_text = qstn_data['question']
        docs = retriever.invoke(qstn_text)
        context =  (' '.join(list(map(lambda d:d.page_content,docs)))).replace('\n', '. ')
        prompts  += [ get_mcq_training_prompt(qstn_data,context)  ]#get_qa_training_prompt(qstn_data,context)

    return prompts


#find the largest token count
def _get_max_length(finetuning_datalist,tokenizer, max_context_length):
    tokens = tokenizer(list(map(lambda e: e['prompt'],finetuning_datalist)),return_tensors='np') #+e['answer']+'\n'+e['explanation']
    argmax_token_len = np.argmax([t.shape[0] for t in tokens.data['input_ids']])
    max_length = tokens.data['input_ids'][argmax_token_len].shape[0]
    max_length = min(max_length, max_context_length)
    return max_length


def _formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = example['prompt'][i] #+ example['answer'][i] + '\n'+example['explanation'][i]
        output_texts.append(text)
    return output_texts


#tokenize data for training
def tokenize_dataset(example, tokenizer, max_length):
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    text = example['prompt'][0] #+ example['answer'][0] + '\n'+example['explanation'][0]
    # print(text)
    tokenized_input = tokenizer(
        text,
        max_length = max_length,
        truncation=True,
        return_tensors="np"
    )
    return tokenized_input

def build_finetine_prompts(
                   llm_name:str,
                   llm_context_length:int,
                   reranker_model,
                   embedding_model_name:str,
                   embedding_model_kwargs:dict,
                   compression_retriever_top_n:int,
                   vectorstore_host:str,
                   vectorstore_port:int,
                   vectorstore_path:str,
                   vectorstore_k:int,
                   training_data_filename:str,
                   prompt_bin_filename:str,
                   dataset_dir:str,
                   n_jobs:int=4
):
    print(f'''
reranker_model              = {reranker_model}
embedding_model_name        = {embedding_model_name}
embedding_model_kwargs      = {embedding_model_kwargs}
compression_retriever_top_n = {compression_retriever_top_n}
vectorstore_host            = {vectorstore_host}
vectorstore_port            = {vectorstore_port}
vectorstore_path            = {vectorstore_path}
vectorstore_k               = {vectorstore_k}
training_data_filename      = {training_data_filename}
prompt_bin_filename         = {prompt_bin_filename}
n_jobs                      = {n_jobs}''')

    data = read_json_file(training_data_filename)
    batch_size = int (len(data)/n_jobs) + 1
    prompts_list_of_list = Parallel(n_jobs=n_jobs)(delayed(_get_prompt)(
        (index % n_jobs),
        list(map(lambda e:e[1],entry)),
               reranker_model,
               embedding_model_name,
               embedding_model_kwargs,
               compression_retriever_top_n,
               vectorstore_path,
               vectorstore_host,
               vectorstore_port,
               vectorstore_k
            ) for index,entry in tqdm(enumerate(break_list_into_chunks(list(data.items()),batch_size))))
    
    finetuning_datalist = flatten_list_of_list(prompts_list_of_list)
    with open(prompt_bin_filename,'wb') as bin_file:
        pickle.dump(finetuning_datalist,bin_file)
        
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    with open(prompt_bin_filename,'rb') as bin_file:
        finetuning_datalist=pickle.load(bin_file)


    print(f"""********************************************************************************
Prompt[0]
********************************************************************************
    {finetuning_datalist[0]['prompt']}""")


    print(f"""********************************************************************************
Prompt[1]
********************************************************************************
    {finetuning_datalist[3]['prompt']}""")

    training_data_length = len(finetuning_datalist)
    print(f'Training data length: {training_data_length}')


    max_length= _get_max_length(finetuning_datalist, tokenizer,llm_context_length)

    print(f'Maximum token length: {max_length}')

    finetuning_dataset = Dataset.from_list(finetuning_datalist)

    tokenized_dataset = finetuning_dataset.map(
        lambda e: tokenize_dataset(e,tokenizer, max_length),
        batched=True,
        batch_size=1,
        drop_last_batch=False
    )

    tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)

    print(split_dataset)

    split_dataset.save_to_disk(dataset_dir)
       

def finetune_model(
                   prompt_bin_filename:str,
                   llm_name:str,
                   llm_context_length:int,
                   output_dir:str,
                   dataset_dir:str,
                   final_model_output_dir:str,
                   num_train_epochs:int = 5,
                   lora_rank:int=16,
                   learning_rate:float=1.0e-3,
                   max_steps:int=1024):
    print(f'''
prompt_bin_filename     = {prompt_bin_filename}
llm_name                = {llm_name}
llm_context_length      = {llm_context_length}
output_dir              = {output_dir}
dataset_dir             = {dataset_dir}
final_model_output_dir  = {final_model_output_dir}
num_train_epochs        = {num_train_epochs}
lora_rank               = {lora_rank}
learning_rate           = {learning_rate}
max_steps               = {max_steps}
''')
     
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    split_dataset = DatasetDict.load_from_disk(dataset_dir)

    base_model = AutoModelForCausalLM.from_pretrained(llm_name,device_map='auto')

    loftq_config = LoftQConfig(loftq_bits=4)  # set 4bit quantization
    lora_config = LoraConfig(
        init_lora_weights="loftq",
        loftq_config=loftq_config,
        r=lora_rank,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    print('Start training')
    
    training_args = TrainingArguments(

    # Learning rate
    learning_rate=learning_rate,

    # Number of training epochs
    num_train_epochs=num_train_epochs,

    # Max steps to train for (each step is a batch of data)
    # Overrides num_train_epochs, if not -1
    max_steps=max_steps,

    # Batch size for training
    per_device_train_batch_size=1,

    # Directory to save model checkpoints
    output_dir=output_dir,

    # Other arguments
    overwrite_output_dir=False, # Overwrite the content of the output directory
    disable_tqdm=False, # Disable progress bars
    eval_steps=64, # Number of update steps between two evaluations
    save_steps=64, # After # steps model is saved
    warmup_steps=1, # Number of warmup steps for learning rate scheduler
    per_device_eval_batch_size=1, # Batch size for evaluation
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    optim="adafactor",
    gradient_accumulation_steps = 4,
    gradient_checkpointing=False,

    # Parameters for early stopping
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    )

    collator = DataCollatorForCompletionOnlyLM(instruction_template=train_instruction_template, response_template=train_response_template, tokenizer=tokenizer)

    # print(split_dataset['train'][0]['prompt'])

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        max_seq_length=llm_context_length,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        data_collator=collator,
        formatting_func=_formatting_prompts_func
    )


    trainer.train()

    peft_model = peft_model.merge_and_unload()
    peft_model.save_pretrained(final_model_output_dir)