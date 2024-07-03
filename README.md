# Folder setup
- `bin` has all the fine-tuned models downloaded from huggingface
- `data` contains the dataset
- `notebooks` containes jupyter notebooks
- `scripts` contains python scripts.
- `lib` contains helper functions.
- `results` contains results in CSV format


The system used had the following specification 
|Spec|Value|
|----|-----|
|RAM | 62 GB|
|Swap| 50 GB|
|CPU| 24|

# Loading data into Vectorstore
Start chromadb service to run on localhost and port 8000 as specified in the configuration. 
```bash
chroma run --path data/vectorstore/chromadb_512_32_all-MiniLM-L6-v2/
```

Execute the driver and specify the `load-docs` subcommand to load all of the documents into the vectorstore.
```bash
python driver.py load-docs  -c config.yaml
```   

# Fintuning the LLM model
