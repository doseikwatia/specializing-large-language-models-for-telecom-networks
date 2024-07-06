# Folder setup
- `bin` has all the fine-tuned models downloaded from huggingface
- `data` contains the dataset
- `notebooks` containes jupyter notebooks
- `scripts` contains python scripts.
- `lib` contains helper functions.
- `results` contains results in CSV format
- `driver.py` is a program that helps execute the various steps from ingesting data to running inference to obtain results.


The system used had the following specification 
|Spec|Value|
|----|-----|
|RAM | 62 GB|
|Swap| 50 GB|
|CPU| 24|
|GPU Count|2|
|GPU RAM|2 x 11.264 GB |

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
## Building the training prompts
Now that the vectorestore has been created we can negate `vectorstore.port` eg `-800` to force the driver to use the persistent storage directly. To fine-tune the model we have to build the training prompt first. Execute the command below to build the prompt file. I had to separate this step from the actual training step in order to clean up and reserve enough memory for the training stage.
```bash
python driver.py build-finetune-prompt
```
## Training the model
With the training prompt binary file  in place execute the following script to mint a `PEFT` modified model. When training set the `common.compression_retriever_top_n` value to `2` in order to fit into limited GPU memeory.
```bash
python driver.py fine-tune-model
```

## Bulding inference prompt
In order to build prompt for inference execute the command below
```bash 
python driver.py build-inference-prompt
```
The `inference.*` section of the configuration determines what the command does. The following run modes are supported.

|Run Mode| Action |
|--------|--------|
| 0      | Only training data|
| 1      | Only testing data|
| 2      | Both training and test data |

## Running inference


# Reproducing Results
Set the following parameters to the values specified at the following stages in order to reproduce the results. Set the following configuration when building prompt and finetuning the model.

|Parameter|Value|
|---------|-----|
|`common.vectorstore.k`|50|
|`common.vectorstore.compression_retriever_top_n`|1|
|`inference.run_mode`|0|

At inference time use the following
|Parameter|Value|
|---------|-----|
|`common.vectorstore.k`|200|
|`common.vectorstore.compression_retriever_top_n`|9|
|`inference.run_mode`|1|