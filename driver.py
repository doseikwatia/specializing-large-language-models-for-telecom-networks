import argparse
import yaml

from lib.constants import SUB_COMD_LOAD_DOC
from lib.data_ingest import load_documents



def main():
    parser = argparse.ArgumentParser(prog='driver')
    subparser = parser.add_subparsers( help='sub commands', dest='subcommand')
    
    #load-docs subcommand
    parser_load_docs = subparser.add_parser(SUB_COMD_LOAD_DOC)
    parser_load_docs.add_argument('-c','--config',default='config.yaml',required=True)
    # parser_load_docs.add_argument('')
    
    args = parser.parse_args()
    
    with open(args.config, mode='r',encoding='UTF-8') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        
    if args.subcommand == SUB_COMD_LOAD_DOC:
        load_documents(
            embedding_model_name=config['common']['embedding_model']['name'],
            embedding_model_kwargs=config['common']['embedding_model']['kwargs'],
            vectorstore_path= config['common']['vectorstore']['path'],
            textsplitter_chunk_size=config['data-ingest']['textsplitter']['chunk_size'],
            textsplitter_overlap=config['data-ingest']['textsplitter']['overlap'],
            documents_path=config['data-ingest']['documents']['path'],
            documents_extentions=config['data-ingest']['documents']['extensions'],
        )


if __name__ == '__main__':
    main()