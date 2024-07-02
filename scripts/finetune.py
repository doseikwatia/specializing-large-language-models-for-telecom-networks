# %%
import os
import os.path as osp
import sys
ROOT_DIR = osp.dirname(os.getcwd())
sys.path.append(ROOT_DIR)

# %%
from transformers import AutoTokenizer
import numpy as np
from datasets import Dataset
from lib.config import LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, VECTOR_STORE_NAME,COMPRESSION_RETRIEVER_TOP_N,VECTOR_RETRIEVER_K,RERANKER_MODEL_NAME,COMPRESSION_RETRIEVER_TOP_N

# %%
CONTEXT_LENGTH = 2048
MAX_STEPS=1024
BATCH_SIZE=512
GENERATE_PROMPTS=False
VECTOR_STORE_PATH = '../data/vectorstore/'
DEVICE_MAP='auto' #{0:'cuda:1',1:'cuda:2'}

print(f""" 
     LLM_MODEL_NAME={LLM_MODEL_NAME}
     EMBEDDING_MODEL_NAME={EMBEDDING_MODEL_NAME}
     RERANKER_MODEL_NAME={RERANKER_MODEL_NAME}
     VECTOR_STORE_NAME={VECTOR_STORE_NAME}
     COMPRESSION_RETRIEVER_TOP_N={COMPRESSION_RETRIEVER_TOP_N}
     VECTOR_RETRIEVER_K={VECTOR_RETRIEVER_K}
     CONTEXT_LENGTH={CONTEXT_LENGTH}
     MAX_STEPS={MAX_STEPS}
     VECTOR_STORE_PATH={VECTOR_STORE_PATH}
     BATCH_SIZE:{BATCH_SIZE}
     GENERATE_PROMPTS:{GENERATE_PROMPTS}
""")
# %%
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
# %% [markdown]
# # Set up vectorstore

# %%
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from lib.normic_wrapper import NomicEmbedding
from tqdm import tqdm

# %%
def create_retriever():
    embeddings  = GPT4AllEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    device='gpu')
    # embeddings = NomicEmbedding(model_name=EMBEDDING_MODEL_NAME,dimensionality=512,device='gpu')
    vectorstore = Chroma(persist_directory=VECTOR_STORE_PATH+VECTOR_STORE_NAME, embedding_function=embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': VECTOR_RETRIEVER_K}
    )
   
    rerank_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME, model_kwargs = {'device': 'cuda'})

    compressor = CrossEncoderReranker(model=rerank_model, top_n=COMPRESSION_RETRIEVER_TOP_N)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

# %% [markdown]
# # Set up database

# %%
import json
from lib.prompt import get_mcq_training_prompt, get_qa_training_prompt
from joblib import Parallel, delayed
import pickle


def read_data(filename):
    with open(filename) as json_file :
        json_data = json.load(json_file)
    return json_data

data = read_data("../data/TeleQnA_training.txt")

def get_prompt(qstn_datas):
    retriever = create_retriever()
    prompts = []
    for qstn_data in tqdm(qstn_datas):
        qstn_text = qstn_data['question']
        docs = retriever.invoke(qstn_text)
        context =  (' '.join(list(map(lambda d:d.page_content,docs)))).replace('\n', '. ')
        prompts  += [get_qa_training_prompt(qstn_data,context), get_mcq_training_prompt(qstn_data,context)]

    return prompts

def chunks(container,size):
    for i in range(0, len(container), size):
        yield container[i:i + size]
        
def flatten(container):
    result = []
    for chunk in container:
        result += chunk
    return result 
           
# finetuning_datalist = list(map(lambda entry:get_prompt(entry[1],create_retriever()),tqdm(data.items())))
if GENERATE_PROMPTS:
    finetuning_datalist = flatten(Parallel(n_jobs=4)(delayed(get_prompt)(list(map(lambda e:e[1],entry))) for entry in tqdm(chunks(list(data.items()),BATCH_SIZE))))
    with open('../bin/pickle/finetuning_datalist.pkl','wb') as bin_file:
        pickle.dump(finetuning_datalist,bin_file)
else:
    with open('../bin/pickle/finetuning_datalist.pkl','rb') as bin_file:
        finetuning_datalist=pickle.load(bin_file)



# %%
print(f"""********************************************************************************
Prompt[0]
********************************************************************************
{finetuning_datalist[0]['prompt']}""")


print(f"""********************************************************************************
Prompt[1]
********************************************************************************
{finetuning_datalist[1]['prompt']}""")

# %%
len(finetuning_datalist)

# %%
#find the largest token count
def get_max_length(finetuning_datalist,tokenizer):
    tokens = tokenizer(list(map(lambda e: e['prompt'],finetuning_datalist)),return_tensors='np') #+e['answer']+'\n'+e['explanation']
    argmax_token_len = np.argmax([t.shape[0] for t in tokens.data['input_ids']])
    max_length = tokens.data['input_ids'][argmax_token_len].shape[0]
    max_length = min(max_length, CONTEXT_LENGTH)
    return max_length

# %%
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

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = example['prompt'][i] #+ example['answer'][i] + '\n'+example['explanation'][i]
        output_texts.append(text)
    return output_texts

# %%
max_length= get_max_length(finetuning_datalist, tokenizer)

# %%
print(f'MAX_LENGTH: {max_length}')

# %%
finetuning_dataset = Dataset.from_list(finetuning_datalist)

# %%
finetuning_dataset

# %%
tokenized_dataset = finetuning_dataset.map(
    lambda e: tokenize_dataset(e,tokenizer, max_length),
    batched=True,
    batch_size=1,
    drop_last_batch=True
)

# %%
tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)

print(split_dataset)

split_dataset.save_to_disk("../data/finetuning/split_dataset")


# %%
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from peft import LoftQConfig, LoraConfig, get_peft_model
from lib.prompt import train_response_template, train_instruction_template, train_explanation_template
from trl import  SFTTrainer, DataCollatorForCompletionOnlyLM 

# %%
training_config = {
    "model": {
        "pretrained_name": LLM_MODEL_NAME,
        "max_length" : CONTEXT_LENGTH
    },
    "datasets": {
        "use_hf": False,
        "path": "../data/finetuning/split_dataset/"
    },
    "verbose": True
}

# %% [markdown]
# ## Load base model

# %%
base_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME,device_map=DEVICE_MAP)

# %%
print(base_model)

# %%
loftq_config = LoftQConfig(loftq_bits=4)           # set 4bit quantization
lora_config = LoraConfig(
    init_lora_weights="loftq",
    loftq_config=loftq_config,
    r=8,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False
)
peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()

# %%
training_args = TrainingArguments(

  # Learning rate
  learning_rate=1.0e-3,

  # Number of training epochs
  num_train_epochs=1,

  # Max steps to train for (each step is a batch of data)
  # Overrides num_train_epochs, if not -1
#   max_steps=MAX_STEPS,

  # Batch size for training
  per_device_train_batch_size=1,

  # Directory to save model checkpoints
  output_dir='../bin/',

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

# %%
collator = DataCollatorForCompletionOnlyLM(instruction_template=train_instruction_template, response_template=train_response_template, tokenizer=tokenizer)

# %%
# print(split_dataset['train'][0]['prompt'])

# %%
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    max_seq_length=max_length,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    data_collator=collator,
    formatting_func=formatting_prompts_func
)

# %%
trainer.train()
# %%
peft_model = peft_model.merge_and_unload()
peft_model.save_pretrained('../bin/pretrained_1024_32')