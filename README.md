# text-generation
> Generation script(s) for generating text with LLMs

## Description
This repository contains scripts for generating text with language models. The scripts are designed to be used with the Hugging Face `transformers` library and the `datasets` library. Accelerate is used to speed up the generation process.

## Requirements

### Dependencies
Install the Python dependencies defined in the requirements.txt.
```bash
pip install -r requirements.txt
```

### Accelerate
Setup accelerate:
```bash
accelerate config
```

### OpenAI API access
To use the OpenAI API, you need to set the `OPENAI_API_KEY` environment variable to your API key.

## Generation with Hugging Face models
The `run_gen.py` script will generate samples from a specified dataset with a Hugging Face model. It supports a large set of options for controlling the generation process.

### Usage

```bash
usage: run_gen.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH] [--model_type MODEL_TYPE] [--config_name CONFIG_NAME] [--tokenizer_name TOKENIZER_NAME]
                  [--use_fast_tokenizer [USE_FAST_TOKENIZER]] [--no_use_fast_tokenizer] [--model_revision MODEL_REVISION] [--token TOKEN] [--use_auth_token [USE_AUTH_TOKEN]]
                  [--dataset_name DATASET_NAME] [--dataset_config_name DATASET_CONFIG_NAME] [--dataset_split DATASET_SPLIT] [--text_column_name TEXT_COLUMN_NAME]
                  [--reference_column_name REFERENCE_COLUMN_NAME] [--dataset_revision DATASET_REVISION] [--streaming [STREAMING]] [--overwrite_cache [OVERWRITE_CACHE]]
                  [--validation_split_percentage VALIDATION_SPLIT_PERCENTAGE] [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS] [--log_preditions [LOG_PREDITIONS]]
                  [--log_predition_samples LOG_PREDITION_SAMPLES] [--generation_config_file GENERATION_CONFIG_FILE] [--per_device_batch_size PER_DEVICE_BATCH_SIZE]
                  [--output_dir OUTPUT_DIR] [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]] [--id_column_name ID_COLUMN_NAME] [--keep_columns KEEP_COLUMNS [KEEP_COLUMNS ...]]
                  [--seed SEED] [--max_new_tokens MAX_NEW_TOKENS] [--max_window_size MAX_WINDOW_SIZE] [--subsamples SUBSAMPLES] [--use_brace_matching [USE_BRACE_MATCHING]]
                  [--brace_matching_start_level BRACE_MATCHING_START_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        The model checkpoint for weights initialization. Do not set if you want to train a model from scratch. (default: None)
  --model_type MODEL_TYPE
                        If training from scratch, pass a model type from the list: bart, bert, bert-generation, big_bird, bigbird_pegasus, biogpt, blenderbot, blenderbot-small,
                        bloom, camembert, llama, codegen, cpmant, ctrl, data2vec-text, electra, ernie, falcon, fuyu, git, gpt2, gpt2, gpt_bigcode, gpt_neo, gpt_neox,
                        gpt_neox_japanese, gptj, llama, marian, mbart, mega, megatron-bert, mistral, mixtral, mpt, musicgen, mvp, open-llama, openai-gpt, opt, pegasus, persimmon,
                        phi, plbart, prophetnet, qdqbert, reformer, rembert, roberta, roberta-prelayernorm, roc_bert, roformer, rwkv, speech_to_text_2, transfo-xl, trocr, whisper,
                        xglm, xlm, xlm-prophetnet, xlm-roberta, xlm-roberta-xl, xlnet, xmod (default: None)
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as model_name (default: None)
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as model_name (default: None)
  --use_fast_tokenizer [USE_FAST_TOKENIZER]
                        Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. (default: True)
  --no_use_fast_tokenizer
                        Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. (default: False)
  --model_revision MODEL_REVISION
                        The specific model version to use (can be a branch name, tag name or commit id). (default: main)
  --token TOKEN         The token to use as HTTP bearer authorization for remote files. If not specified, will use the token generated when running `huggingface-cli login` (stored in
                        `~/.huggingface`). (default: None)
  --use_auth_token [USE_AUTH_TOKEN]
                        The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead. (default: None)
  --dataset_name DATASET_NAME
                        The name of the dataset to use (via the datasets library). (default: None)
  --dataset_config_name DATASET_CONFIG_NAME
                        The configuration name of the dataset to use (via the datasets library). (default: None)
  --dataset_split DATASET_SPLIT
                        The dataset split to use. (default: None)
  --text_column_name TEXT_COLUMN_NAME
                        The dataset column name to use. (default: None)
  --reference_column_name REFERENCE_COLUMN_NAME
                        The dataset column name to use as reference for the target sequence. (default: None)
  --dataset_revision DATASET_REVISION
                        The specific dataset version to use (can be a branch name, tag name or commit id). (default: main)
  --streaming [STREAMING]
                        Enable streaming mode (default: False)
  --overwrite_cache [OVERWRITE_CACHE]
                        Overwrite the cached training and evaluation sets (default: False)
  --validation_split_percentage VALIDATION_SPLIT_PERCENTAGE
                        The percentage of the train set used as validation set in case there is no validation split (default: 5)
  --preprocessing_num_workers PREPROCESSING_NUM_WORKERS
                        The number of processes to use for the preprocessing. (default: None)
  --log_preditions [LOG_PREDITIONS]
                        Whether to log predictions during training. (default: False)
  --log_predition_samples LOG_PREDITION_SAMPLES
                        Number of samples to log during training. (default: 10)
  --generation_config_file GENERATION_CONFIG_FILE
                        Generation config path if not the same as model_name. (default: None)
  --per_device_batch_size PER_DEVICE_BATCH_SIZE
                        Batch size (per device) for generation. (default: 8)
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written. (default: None)
  --overwrite_output_dir [OVERWRITE_OUTPUT_DIR]
                        Overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory. (default: False)
  --id_column_name ID_COLUMN_NAME
                        The column name of the dataset to use as id. If not provided, the index will be used. (default: None)
  --keep_columns KEEP_COLUMNS [KEEP_COLUMNS ...]
                        The column names of the dataset to keep separate by commas. If not provided, all columns will be removed. (default: None)
  --seed SEED           Seed for random number generation. (default: None)
  --max_new_tokens MAX_NEW_TOKENS
                        The maximum number of new tokens to generate. (default: None)
  --max_window_size MAX_WINDOW_SIZE
                        The maximum number of tokens in the input. (default: None)
  --subsamples SUBSAMPLES
                        The number of subsamples to use from each data example. Randomly selected. None means use all. (default: None)
  --use_brace_matching [USE_BRACE_MATCHING]
                        Whether to use brace matching as a stopping criteria. (default: False)
  --brace_matching_start_level BRACE_MATCHING_START_LEVEL
                        The level of brace matching to start from. (default: 0)
```

### Complete generation
Complete generation is done by providing both an input data column and a reference data column. This will make the model use the whole prompt as input. By setting `--max_new_tokens` to `auto`, all the unused embedding space is used to generate as much as possible. The maximum number of tokens in the input can be truncated (from the left) by setting `--max_window_size`, thus allowing for a longer output (`--max_new_tokens`).

#### Example
The following example will generate samples from the test split of the [methods2test_small](https://huggingface.co/datasets/andstor/methods2test_small) dataset using the greedy decoding strategy. The output will be saved to the `output` directory.

```bash
accelerate launch run_gen.py \
--dataset_name andstor/methods2test_small \
--dataset_config_name fm+fc+c+m+f+t+tc \
--dataset_split test \
--text_column_name source \
--reference_column_name target \
--model_name_or_path facebook/opt-125m \
--per_device_batch_size 4 \
--output_dir output \
--overwrite_output_dir \
--seed 42 \
--preprocessing_num_workers 10 \
--max_new_tokens auto
```

#### Early stopping
Early stopping can be done by using brace matching. This will stop the generation when the number of open braces is equal to the number of closed braces. The level of brace matching to start from can be controlled by the `--brace_matching_start_level` argument. The following example will use brace matching as a stopping criteria and start from level 1.

```bash
--use_brace_matching \
--brace_matching_start_level 1
```

### Strided generation
Stridden generation is done by only providing an input data column. This will be split into parts where each part is generated in a "sliding window" approach. Each window serves as the reference for the preceding window. The window stride (read size) is determined by the `--max_window_size` argument. The number of tokens to be generated is controlled by `--max_new_tokens`. The number of subsamples to use from each data example can be controlled by `--subsamples`. If `--subsamples` is set to `None`, all subsamples will be used.

#### Filtering
The data is filtered by the following criteria:
- The input is at least max_new_tokens + max_window_size long

#### Window splitting
Given an input, it is first truncated by max_new_tokens. The result is then split into window sizes of up to max_window_size. The minimum size of each window is l / math.ceil(l / max_size), where l is len(imput)-max_new_tokens. Each window is generated independently. The first window has an index of 0, the second has an index of 1, etc.

#### Example
The following example will generate samples from the test split of the [The Pile](https://pile.eleuther.ai/) dataset using the greedy decoding strategy. The input will be truncated to 512 tokens and the maximum number of new tokens will be 512. The output will be saved to the `output` directory.


```bash
accelerate launch generate.py \
--dataset_name andstor/the_pile_github \
--dataset_config_name java \
--dataset_split test \
--text_column_name text \
--model_name_or_path EleutherAI/gpt-j-6B \
--generation_config_file generation_config.json \
--per_device_batch_size 1 \
--output_dir output \
--seed 42 \
--preprocessing_num_workers 20 \
--max_window_size 512 \
--max_new_tokens 512
```

The `generation_config.json` file contains the following configuration:
```json
{
    "do_sample": false,
    "max_new_tokens": 256,
    "bos_token_id": 50256,
    "eos_token_id": 50256
}
```


## Generation with third-party models (API)
Several third-party models such as ChatGPT have been used for generating data. Scripts for generating data with these models can be found in the `notebooks` directory. Note that access to most of these requires a paid subscription. Furthermore, most are closed-source models and might not be reproducible. 


## License

Copyright © [André Storhaug](https://github.com/andstor)

This repository is licensed under the [MIT License](https://github.com/andstor/verified-smart-contracts/blob/main/LICENSE).
