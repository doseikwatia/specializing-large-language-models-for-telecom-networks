import argparse
import yaml

from lib.constants import SUB_COMD_LOAD_DOC



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
        configuration = {
        'embedding_model' : config['common']['embedding_model'],
        'vectorstore'  : config['common']['vectorstore'],
        'ingest_info' : config['data-ingest'],
        }
        load_documents(configuration)


if __name__ == '__main__':
    main()