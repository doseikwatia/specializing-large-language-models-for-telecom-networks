import argparse
import yaml

from lib.constants import SUB_COMD_BUILD_FINETUNE_PROMPT, SUB_COMD_FINE_TUNE, SUB_COMD_LOAD_DOC
from lib.data_ingest import load_documents
from lib.finetune import build_finetine_prompts, finetune_model



def main():
    parser = argparse.ArgumentParser(prog='driver')
    subparser = parser.add_subparsers( help='sub commands', dest='subcommand')
    
    #load-docs subcommand
    parser_load_docs = subparser.add_parser(SUB_COMD_LOAD_DOC)
    parser_load_docs.add_argument('-c','--config',default='config.yaml',required=False)
    # parser_load_docs.add_argument('')
    
    # build-finetune-prompt
    parser_build_finetune_prompt = subparser.add_parser(SUB_COMD_BUILD_FINETUNE_PROMPT)
    parser_build_finetune_prompt.add_argument('-c','--config',default='config.yaml',required=False)
    # fine-tune
    parser_finetune = subparser.add_parser(SUB_COMD_FINE_TUNE)
    parser_finetune.add_argument('-c','--config',default='config.yaml',required=False)
    
    args = parser.parse_args()
    
    with open(args.config, mode='r',encoding='UTF-8') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        
    if args.subcommand == SUB_COMD_LOAD_DOC:
        load_documents(
            embedding_model_name=config['common']['embedding_model']['name'],
            embedding_model_kwargs=config['common']['embedding_model']['kwargs'],
            vectorstore_path= config['common']['vectorstore']['path'],
            vectorstore_host= config['common']['vectorstore']['host'],
            vectorstore_port= config['common']['vectorstore']['port'],
            textsplitter_chunk_size=config['data-ingest']['textsplitter']['chunk_size'],
            textsplitter_overlap=config['data-ingest']['textsplitter']['overlap'],
            documents_path=config['data-ingest']['documents']['path'],
            documents_extentions=config['data-ingest']['documents']['extensions'],
            n_jobs=config['data-ingest']['n_jobs'],
            index_chunk=config['data-ingest']['index_chunk']
        )
    elif args.subcommand == SUB_COMD_BUILD_FINETUNE_PROMPT:
        build_finetine_prompts(
            reranker_model              = config['common']['reranker_model'],
            embedding_model_name        = config['common']['embedding_model']['name'],
            embedding_model_kwargs      = config['common']['embedding_model']['kwargs'],
            compression_retriever_top_n = config['common']['compression_retriever_top_n'],
            vectorstore_host            = config['common']['vectorstore']['host'],
            vectorstore_port            = config['common']['vectorstore']['port'],
            vectorstore_path            = config['common']['vectorstore']['path'],
            vectorstore_k               = config['common']['vectorstore']['k'],
            training_data_filename      = config['training']['data_filename'],
            prompt_bin_filename         = config['training']['prompt_bin_filename'],
            n_jobs                      = config['training']['n_jobs'],
            llm_context_length          = config['common']['llm_model']['context_length'],
            llm_name                    = config['common']['llm_model']['name'],
            dataset_dir                 = config['training']['dataset_dir'],
        )
    elif args.subcommand == SUB_COMD_FINE_TUNE:
        finetune_model(
            prompt_bin_filename         = config['training']['prompt_bin_filename'],
            llm_name                    = config['common']['llm_model']['name'],
            llm_context_length          = config['common']['llm_model']['context_length'],
            output_dir                  = config['training']['output_dir'],
            dataset_dir                 = config['training']['dataset_dir'],
            final_model_output_dir      = config['common']['trained_model_dir'],
            num_train_epochs            = config['training']['epochs'],
            lora_rank                   = config['training']['lora_rank'],
            learning_rate               = config['training']['learning_rate'],
        )
        
        
    print('driver exiting')

if __name__ == '__main__':
    main()