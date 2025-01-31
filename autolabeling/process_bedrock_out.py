import json
import os
from dataset_utils import get_qa_dataset
from collections import defaultdict
from tqdm import tqdm
import re
import yaml
import argparse
from easydict import EasyDict as edict


def extract_answer(text):
    pattern = r"answer is (\(?([A-J])\)?|\\?boxed\{(.*?)\})" # second part is only for non multiple choice
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return None

def get_data(data_dir):
    '''
    aggregates all jsonl file in directory into one list
    '''
    data = []
    for filename in os.listdir(data_dir):

        if filename.endswith('.jsonl.out') or filename.endswith('.jsonl'):
        # if filename.endswith('.jsonl'):


            file_path = f'{data_dir}/{filename}'
            with open(file_path, 'r') as f:
                for l in f:
                    data.append(json.loads(l))

        elif filename.endswith('json'):
            file_path = f'{data_dir}/{filename}'
            with open(file_path, 'r') as f:
                data.extend(json.load(f))

    return data

def dataset_to_id_dict(ds):

    id_dict = {}

    for d in ds:
        
        # add in chain of thoughts key
        d['chain_of_thoughts'] = []
        id_dict[d['id']] = d

    return id_dict

def get_datasets_by_id(data_splits):
    '''
    dataset
        split
            dictionary (id, data)
    '''


    datasets_by_id = defaultdict(dict)

    for d in data_splits:
        dataset = d['dataset']

        subsample = d['subsample'] if 'subsample' in d else None
        subsample_seed = d['subsample_seed'] if 'subsample_seed' in d else None

        for split in d['splits']:
            ds = get_qa_dataset(dataset=dataset, split=split, subsample=subsample, subsample_seed=subsample_seed)
            id_dict = dataset_to_id_dict(ds)

            datasets_by_id[dataset][split] = id_dict


    return datasets_by_id

def parse_bedrock_id(recordId):

    id, cot_id = recordId.split('+')
    dataset, split = id.split('_')[1:]

    return id, cot_id, dataset, split

def parse_cot(cot_str, delimiter='\n\n'):
    '''
        splits cot string into list
        parses out the generated answer
    '''

    cot_splitted = cot_str.strip().split(delimiter)
    parsed_answer = extract_answer(cot_splitted[-1])
    return cot_splitted, parsed_answer


def flatten_datasets_by_id(datasets_by_id, dump=True, dump_dir='.'):

    for dataset in datasets_by_id:
        for split in datasets_by_id[dataset]:

            datasets_by_id[dataset][split] =list(datasets_by_id[dataset][split].values())

            if dump:
                with open(f'{dump_dir}/{dataset}_{split}.json', 'w') as f:
                    json.dump(datasets_by_id[dataset][split], f, indent=2)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Script to proccess Bedrock output files.')
    parser.add_argument('-c','--config', type=str, help='Path to config json', default='./gen_cot_configs/sciqqa_trainval_32.yml')
    args = parser.parse_args()

    with open(args.config) as stream:
        try:
            configs = edict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)


    output_directory = f'./bedrock_outputs/{configs.experiment_name}'

    bedrock_data = get_data(output_directory)
    datasets_by_id = get_datasets_by_id(configs.data_splits)

    for d in tqdm(bedrock_data):

        id, cot_id, dataset, split = parse_bedrock_id(d['recordId'])
        # if 'chain_of_thoughts' not in datasets_by_id[dataset][split][id]:
        #    datasets_by_id[dataset][split][id]['chain_of_thoughts'] = []

        stop_reason = d['modelOutput']['stop_reason']

        # we want to ignore bad CoT
        if stop_reason == 'length':
            continue

        steps, parsed_answer = parse_cot(d['modelOutput']['generation'])
        if parsed_answer == None and ('no_parse' not in configs or configs['no_parse']==False): 
            # again to ignore bad CoT
            # but only if no_parse is false
            continue



        datasets_by_id[dataset][split][id]['chain_of_thoughts'].append({'steps':steps, 
                                            'parsed_answer':parsed_answer,
                                            'parsed_answer_correctness': parsed_answer==datasets_by_id[dataset][split][id]['answer'],
                                            'cot_id':cot_id})
        
    cot_output_path = f'cot_data/{configs.experiment_name}/'
    os.makedirs(os.path.dirname(cot_output_path), exist_ok=True)

    flatten_datasets_by_id(datasets_by_id, dump=True, dump_dir=cot_output_path)

# TODO: add in cot id (do this in the fix part)
# 
# Put jupyternotebook stuff in here (add polish e.g. directory stuff)
# work on script to prepare math shephard augs
    # id selection
        # compute proportion correct
            # ensure at least 16 cot
            # correct proportion is less than 80%?
        # keep these ids

    # cot selection
        # for each selected id,
            # take every incorrect cot
            # and remaining are correct (up to 16)?
            # maybe we want to keep bedrock inputs? (so that we can directly prepare new input)
                # maybe not
        # output final selected as json file
    # prepare bedrock


# converts data into following json format

# question
# answer
# id
# metadata
# chain_of_thoughts (list)
    # steps
    # labels
    # augs
    # parsed_answer
    # parsed_answer_correctness


######

# majority vote
# best of n


# function that iterates through chain of thought
    # specify what we want
        # list of lambda functions on chain of thought
          # correctness (just return partial correctness)
          # aggregations
