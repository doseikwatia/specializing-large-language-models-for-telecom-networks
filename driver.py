import argparse
import yaml

from lib.constants import SUB_COMD_BUILD_FINETUNE_PROMPT, SUB_COMD_FINE_TUNE, SUB_COMD_LOAD_DOC, SUB_COMDS_BUILD_INFERENCE_PROMPT, SUB_COMDS_TEST_MODEL, SUB_COMDS_METRIC
from lib.data_ingest import load_documents
from lib.finetune import build_finetine_prompts, finetune_model
from lib.inference import build_inference_prompt, calculate_accuracy, test_model


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
    
    # build inference prompt
    parser_build_inference_prompt = subparser.add_parser(SUB_COMDS_BUILD_INFERENCE_PROMPT)
    parser_build_inference_prompt.add_argument('-c','--config',default='config.yaml',required=False)
    
    # test model
    parser_test_model = subparser.add_parser(SUB_COMDS_TEST_MODEL)
    parser_test_model.add_argument('-c','--config',default='config.yaml',required=False)
    
    # metrics
    parser_test_model = subparser.add_parser(SUB_COMDS_METRIC)
    parser_test_model.add_argument('-c','--config',default='config.yaml',required=False)
    
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
            max_steps                   = config['training']['max_steps'],
        )
    elif args.subcommand == SUB_COMDS_BUILD_INFERENCE_PROMPT:
        build_inference_prompt(
            run_mode                     = config['inference']['run_mode'],
            training_input_filename      = config['inference']['training_input_filename'],
            testing_input_filename       = config['inference']['testing_input_filename'],
            embedding_model_name         = config['common']['embedding_model']['name'],
            embedding_model_kwargs       = config['common']['embedding_model']['kwargs'],
            vectorstore_path             = config['common']['vectorstore']['path'],
            vectorstore_host             = config['common']['vectorstore']['host'],
            vectorstore_port             = config['common']['vectorstore']['port'],
            reranker_model               = config['common']['reranker_model'],
            compression_retriever_top_n  = config['common']['compression_retriever_top_n'],
            vectorstore_k                = config['common']['vectorstore']['k'],
            training_prompt_bin_filename = config['inference']['training_prompt_bin_filename'],
            testing_prompt_bin_filename  = config['inference']['testing_prompt_bin_filename'],
            n_jobs                       = config['inference']['n_jobs'],
        )
    elif args.subcommand == SUB_COMDS_TEST_MODEL:
        test_model (
            task                        = config['inference']['task'],
            run_mode                    = config['inference']['run_mode'],
            llm_name                    = config['common']['llm_model']['name'],
            final_model_output_dir      = config['common']['trained_model_dir'],
            training_output_filename    = config['inference']['training_output_filename'],
            testing_output_filename     = config['inference']['testing_output_filename'],
            training_prompt_bin_filename = config['inference']['training_prompt_bin_filename'],
            testing_prompt_bin_filename  = config['inference']['testing_prompt_bin_filename'],
        )
    elif args.subcommand == SUB_COMDS_METRIC:
        calculate_accuracy(
            training_output_filename = config['inference']['training_output_filename'],
            training_results = config['inference']['training_results'],
        )
        
    print('driver exiting')

if __name__ == '__main__':
    main()