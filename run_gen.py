import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, GenerationConfig, AutoConfig
from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch.utils.data import DataLoader
import json
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    StoppingCriteriaList,
)

import numpy as np
import logging
import random
import warnings
import sys
import os
from arguments import ModelArguments, DatasetArguments, GenerationArguments
from stopping_criterias import BraceMatchingCriteria


from accelerate.logging import get_logger
get_logger("transformers").setLevel(logging.ERROR)
logger = get_logger(__name__)




def main():
    """
    Generate new data by sampling from the original data.
    """
    # See all possible arguments in arguments/*.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DatasetArguments, GenerationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, gen_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, gen_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Initialize accelerator
    accelerator = Accelerator()

    if gen_args.seed is not None:
        set_seed(gen_args.seed)


    # Write the generation config to disk
    if accelerator.is_main_process:
        if os.path.isdir(gen_args.output_dir) and not gen_args.overwrite_output_dir:
            if len(os.listdir(gen_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({gen_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
        
        if gen_args.output_dir is not None:    
            #safe_dataset_name = urllib.parse.quote(args.dataset_name, safe='')
            #urlencode args.dataset_name
            #safe_model_name = urllib.parse.quote(args.model_name_or_path, safe='')
            Path(gen_args.output_dir).mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("Need a output directory.")
    accelerator.wait_for_everyone()


    #if accelerator.is_main_process:
    #    # write args to disk
    #    with open( save_dir / "args.json", "w") as f:
    #        json.dump(args.__dict__, f, indent=4)
    

        # 
    # Load the dataset
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # if args.dataset_name is not None:
    #    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    #    dataset = raw_datasets.with_format("torch", columns=[text_column])

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.dataset_split)#, revison=data_args.dataset_revision)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, revision=model_args.model_revision)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, use_fast=model_args.use_fast_tokenizer, revision=model_args.model_revision, padding_side='left')
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer, revision=model_args.model_revision, padding_side='left')
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token = tokenizer.eos_token

    if accelerator.distributed_type == DistributedType.NO \
       or (accelerator.distributed_type == DistributedType.MULTI_GPU and accelerator.num_processes <= 1):
        device_map = "auto" # Activate the naive model parallelism.
    else:
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        revision=model_args.model_revision,
        device_map=device_map,
    )
    model.tie_weights()
    
    if model_args.adapter_name_or_path is not None:
        model.load_adapter(model_args.adapter_name_or_path)

    generation_config = {}
    if gen_args.generation_config_file is not None:
        # read from file
        with open(gen_args.generation_config_file, "r") as f:
            generation_config = json.load(f)
            generation_config = GenerationConfig.from_dict(generation_config)


    elif model_args.model_name_or_path:
        generation_config = model.generation_config
        logger.warning(f"Using default generation config from model: {generation_config}")

    if gen_args.max_new_tokens is not None:

        if type(gen_args.max_new_tokens) == int:
            generation_config.max_new_tokens = gen_args.max_new_tokens
        elif gen_args.max_new_tokens == "auto":
            if data_args.reference_column_name is not None:
                generation_config.max_new_tokens = None
            else:
                raise ValueError("Automatically determining max_new_tokens is only supported when reference_column_name is set.")
    
        logger.info(f"max_new_tokens are set to {gen_args.max_new_tokens}")
    else:
        raise ValueError("max_new_tokens is not set.")

    if gen_args.max_window_size is None:
        gen_args.max_window_size = model.config.max_position_embeddings - int(generation_config.max_new_tokens or 0)


    # Define stopping criterias
    stopping_criteria_list = StoppingCriteriaList()
    if gen_args.use_brace_matching:
        # Expensive to initialize, so reuse the same instance.
        stopping_criteria_list.append(BraceMatchingCriteria(tokenizer, gen_args.brace_matching_start_level))
    

    # Write the model config and generation config to disk
    if accelerator.is_main_process:
        print(generation_config)

        # Dump the model config without defaults to disk
        with open( Path(gen_args.output_dir) / "model_config_diff.json", "w") as f:
            json.dump(config.to_diff_dict(), f, indent=4)

        # Dump the model config with defaults to disk
        with open(Path(gen_args.output_dir) / "model_config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=4)

        # Dump the generation config without defaults to disk
        with open(Path(gen_args.output_dir) / "generation_config_diff.json", "w") as f:
            json.dump(generation_config.to_diff_dict(), f, indent=4)

        # Dump the generation config with defaults to disk
        with open(Path(gen_args.output_dir) / "generation_config.json", "w") as f:
            json.dump(generation_config.to_dict(), f, indent=4)


    # Preprocessing the datasets.
    column_names = raw_dataset.column_names
    keep_columns = []
    if gen_args.keep_columns is not None:
        keep_columns = gen_args.keep_columns

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    else:
        text_column_name = "text" if "text" in column_names else column_names[0]
        logger.warning(f"Using column {text_column_name} as text column.")

   
    if data_args.reference_column_name is not None:
        reference_column_name = data_args.reference_column_name
        keep_columns.append(reference_column_name)
        min_input_length = 0
    else:
        min_input_length = gen_args.max_window_size + generation_config.max_new_tokens # TODO: check if this is a good value
        max_input_length = gen_args.max_window_size + generation_config.max_new_tokens
        if max_input_length > model.config.max_position_embeddings:
            raise ValueError(
                f"max_window_size ({gen_args.max_window_size}) + max_new_tokens ({gen_args.max_new_tokens}) is larger than the maximum position embedding size "
                f"({model.config.max_position_embeddings})."
            )

    
    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    
    def filter_function(examples):
        res = []
        is_dropped = 0
        for example in examples["input_ids"]:
            if len(example) < min_input_length:
                res.append(False)
                is_dropped += 1
            else:
                res.append(True)
        if is_dropped:
            logger.info(f"Dropped {is_dropped} examples because they were shorter than {min_input_length} tokens.")
        return res

    def array_chunk_max(array, max_size):
        """
        Splits an array into chunks of maximum size max_size.
        The minimum size of the resulting chunks is l / math.ceil(l / max_size).
        """
        chunks, remainder = divmod(len(array), max_size)
        if remainder > 0:
            chunks += 1
        return np.array_split(array, chunks)



    def cut_function(examples, indices):
        new_examples = {
            "id": [],
            "part": [],
            "input_ids": [],
            "attention_mask": [],
            "reference_input_ids": [],
        }

        for i, id in enumerate(indices):
            if gen_args.id_column_name is not None:
                id = examples[gen_args.id_column_name][i]

            input_ids = examples["input_ids"][i][:-generation_config.max_new_tokens]
            mask = examples["attention_mask"][i][:-generation_config.max_new_tokens]
            
            minibatch_ids = array_chunk_max(input_ids, gen_args.max_window_size)
            minibatch_mask = array_chunk_max(mask, gen_args.max_window_size)
            
            reference_input_ids = minibatch_ids[1:]
            end_ids = examples["input_ids"][i][-generation_config.max_new_tokens:]
            reference_input_ids.append(end_ids)

            minibatch_size = len(minibatch_ids)
            sample_size = minibatch_size
            if gen_args.subsamples is not None:
                sample_size = min(gen_args.subsamples, minibatch_size)

            sample_indices = sorted(random.sample(range(minibatch_size), sample_size))
            minibatch_ids = [minibatch_ids[i] for i in sample_indices]
            minibatch_mask = [minibatch_mask[i] for i in sample_indices]
            reference_input_ids = [reference_input_ids[i] for i in sample_indices]
            

            new_examples["id"].extend([id]*sample_size)
            new_examples["part"].extend(list(zip(sample_indices, [minibatch_size]*sample_size)))
            new_examples["input_ids"].extend(minibatch_ids)
            new_examples["attention_mask"].extend(minibatch_mask)
            new_examples["reference_input_ids"].extend(reference_input_ids)
        return new_examples


    def single_batch_function(examples, indices):
        new_examples = {
            "id": [],
            "part": [],
            "input_ids": [],
            "attention_mask": [],
        }

        for i, id in enumerate(indices):
            if gen_args.id_column_name is not None:
                id = examples[gen_args.id_column_name][i]

            input_ids = examples["input_ids"][i][-gen_args.max_window_size:]
            mask = examples["attention_mask"][i][-gen_args.max_window_size:]

            new_examples["id"].append(id)
            new_examples["part"].append([1,1])
            new_examples["input_ids"].append(input_ids)
            new_examples["attention_mask"].append(mask)
        return new_examples


    with accelerator.main_process_first():
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        filtered_dataset = tokenized_dataset.filter(
            filter_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Filtering min length",
        )
        minibatch_dataset = filtered_dataset.map(
            cut_function if reference_column_name is None else single_batch_function,
            with_indices=True,
            batched=True,
            remove_columns=[c for c in filtered_dataset.column_names if c not in keep_columns],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Preparing minibatches",
        )
        #dataset = tokenized_datasets.with_format("torch", columns=[text_column], output_all_columns=True)
    
    dataset = minibatch_dataset
    
    def data_collator(examples):
        batch = tokenizer.pad(examples)
        batch["input_ids"] = torch.tensor(batch["input_ids"])
        batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        return batch

    # Create the DataLoader
    data_loader = DataLoader(dataset, shuffle=False,
                             collate_fn=data_collator, batch_size=gen_args.per_device_batch_size)

    # Prepare everything with `accelerator`.
    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()
    #model = accelerator.unwrap_model(model)
    #model, data_loader = accelerator.prepare(
    #    model, data_loader, device_placement=[True, False])

    # save the data
    i = "{:05n}".format(accelerator.process_index + 1)
    n = "{:05n}".format(accelerator.num_processes)

    path = Path(gen_args.output_dir) / (f"{i}-of-{n}" + f".{data_args.dataset_split}.jsonl")
    fp = open(path, 'w')

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(data_loader) * gen_args.per_device_batch_size), position=accelerator.process_index) #,disable=not accelerator.is_local_main_process)
    for batch in data_loader:
        prompt_ids = batch["input_ids"]
        prompt_ids.to(accelerator.device)
        attention_mask = batch["attention_mask"]
        attention_mask.to(accelerator.device)

        if generation_config.max_new_tokens is None:
            max_new_tokens = model.config.max_position_embeddings - prompt_ids.shape[-1]
        else:
            max_new_tokens = generation_config.max_new_tokens
        #accelerator.print("Generating...")
        #generation_config.num_return_sequences = 2
        with torch.no_grad():
            # generate the data
            generated = accelerator.unwrap_model(model).generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria_list,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False, #TODO: turn off !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            )
        # decode the data

        decoded_prompts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        
        #predicted_ids = generated[:, -max_new_tokens:]
        predicted_ids = generated[:, prompt_ids.shape[-1]:]
        decoded_predictions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        
        if reference_column_name is None:
            reference_ids = batch["reference_input_ids"]
            decoded_reference = tokenizer.batch_decode(reference_ids, skip_special_tokens=True)
        else:
            decoded_reference = batch[reference_column_name]

        progress_bar.update(gen_args.per_device_batch_size)

        # save the data to disk
        for index in range(generated.shape[0]):
            b_index = index//generation_config.num_return_sequences # original batch index
            # print("saving..."):
            #colnames = batch.keys()
            #entry = {colname: batch[colname][index] for colname in colnames}
            #entry = {"text": batch[text_column_name][index]}
            #entry.pop('input_ids', None)
            #entry.pop('attention_mask', None)
            
            entry = {}
            entry["id"] = batch["id"][b_index]
            entry["part"] = batch["part"][b_index]
            entry["seq"] = [(index % generation_config.num_return_sequences) + 1, generation_config.num_return_sequences]
            entry["prompt"] = decoded_prompts[b_index]
            entry["reference"] = decoded_reference[b_index]
            entry["prediction"] = decoded_predictions[index]

            ended = None
            if len(predicted_ids[index]) == generation_config.max_new_tokens:
                ended = "length"
            for stopping_criteria in stopping_criteria_list: # possible race condition
                if stopping_criteria.is_sequence_stopped(index):
                    ended = stopping_criteria.stop_reason()
            if predicted_ids[index][-1].item() == tokenizer.eos_token_id:
                ended = "stop"
            entry["finish_reason"] = ended

            entry["meta"] = { "subset": data_args.dataset_config_name }

            # keep all "keep_columns":
            if gen_args.keep_columns is not None:
                for colname in gen_args.keep_columns:
                    entry[colname] = batch[colname][b_index]

            fp.write(json.dumps(entry) + "\n")
            fp.flush()
        
        stopping_criteria_list[0].reset() # reset the BraceMatchingCriteria


    fp.close()

    accelerator.wait_for_everyone()
    
    progress_bar.close()


if __name__ == "__main__":
    main()