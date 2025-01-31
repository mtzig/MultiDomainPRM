import json
from tqdm import tqdm
from torch.utils.data import Dataset
from copy import deepcopy


def merge_dicts(dict_list):
    merged_dict = deepcopy(dict_list[0])
    for d in dict_list[1:]:
        for key, value in d.items():
            merged_dict[key].extend(value)
    return merged_dict

def tokenize_step(cot_step, label, tokenizer, label_mask_token_id=-100, label_last_n=None):
    cot_step_tokenized = tokenizer(cot_step, add_special_tokens=False)

    if label_last_n is None:
        cot_step_labels = [label]* len(cot_step_tokenized.input_ids)
    else:
        if  label_last_n > len(cot_step_tokenized.input_ids):
            cot_step_labels = [label]*len(cot_step_tokenized.input_ids)
        else:
            cot_step_labels = [label_mask_token_id]*(len(cot_step_tokenized.input_ids)-label_last_n) + [label]*label_last_n
    
    cot_step_tokenized['labels'] = cot_step_labels

    return cot_step_tokenized


def tokenize_one_cot(question_tokenized, data, tokenizer, label_mask_token_id=-100, label_last_n=None, max_length=None, contrastive=False, use_augs=True, mask_neg=False, skip_neg=False):
    '''
    contrastive: argument to test contrastive lose
        Currently not implemented (so does nothing)

    mask_neg: for chain of thought with negative step, should we mask steps before it
    '''

    if 'labels' not in data:
        return []
    


    labels = data['labels']

    # labels is None if autolabeling failed
    if labels is None:
        return []
    
    # for if we only want to use autolabeled correct CoT
    # and counertfactual augmented steps
    if skip_neg and not all(label==1 for label in labels):
        return []

    cot_steps_tokenized = []

    # if steps are all correct, don't care about masking
    if mask_neg and all(label==1 for label in labels):
            mask_neg = False

    for i,step in enumerate(data['steps']):
        cot_step = f'{step} \n\n\n\n'

        if not mask_neg:
            label = 1 if labels[i] == 1 else 0
        else:
            label = label_mask_token_id if labels[i] == 1 else 0

        cot_step_tokenized = tokenize_step(cot_step, label=label, tokenizer=tokenizer, label_mask_token_id=label_mask_token_id, label_last_n=label_last_n)


        cot_steps_tokenized.append(cot_step_tokenized)

        # for incorrect cot, we want to stop after the first incorrect step
        if label == 0:
            break


    augs = []
    if use_augs:
        for aug in data['augs']:
            aug_idx = aug['aug_idx']
            aug_step_content = aug['aug_step']
            aug_step = f'{aug_step_content} \n\n\n\n'

            
            # if aug['aug_type'] == 1:
            #     continue

            # all augments are incorrect step, except those of type 1 (good) or 0 (okay)
            aug_label = 1 if aug['aug_type'] == 1 or aug['aug_type'] == 0 else 0

            aug_step_tokenized = tokenize_step(aug_step, label=aug_label, tokenizer=tokenizer, label_mask_token_id=label_mask_token_id, label_last_n=label_last_n)
            augs.append((aug_step_tokenized, aug_idx))
 
    tokenized = []

    # chosen_tokenized, is the original full cot that was the chosen completion, from which alternate completions are generated to augment
    chosen_tokenized = merge_dicts([question_tokenized] + cot_steps_tokenized)
    if max_length is None or len(chosen_tokenized.input_ids) <= max_length:
        tokenized.append(chosen_tokenized)

    # we now change all the labels to masks
    for cot_step_tokenized in cot_steps_tokenized:
        cot_step_tokenized['labels'] = [label_mask_token_id] * len(cot_step_tokenized['labels'])

    for aug in augs:
        aug_step_tokenized, aug_idx = aug
        aug_tokenized = merge_dicts([question_tokenized] + cot_steps_tokenized[:aug_idx] + [aug_step_tokenized])

        if max_length is None or len(aug_tokenized.input_ids) <= max_length:
            tokenized.append(aug_tokenized)

    return tokenized

def tokenize_one_question(data, tokenizer, label_mask_token_id=-100, label_last_n=None, max_length=None, contrastive=False, use_augs=True, mask_neg=False, skip_neg=False):
    '''
    can add aug_type param to specify which type of augmentation to use
    '''

    question = data['question']

    question_tokenized = tokenizer(f'{question} \n\n')


    question_tokenized['labels'] = [label_mask_token_id] * len(question_tokenized.input_ids)
    

    tokenized = []

    for cot in data['chain_of_thoughts']:
        tokenized.extend(tokenize_one_cot(question_tokenized, cot, tokenizer, label_mask_token_id, label_last_n, max_length, contrastive, use_augs, mask_neg=mask_neg, skip_neg=skip_neg))
    
    return tokenized


def read_json(d):
    if d.endswith('jsonl'):
        text_data = []
        with open(d, 'r') as f:
            for line in f:
                text_data.append(json.loads(line))
    elif d.endswith('json'):
        with open(d, 'r') as f:
            text_data = (json.load(f))
    else:
        raise NotImplementedError('currently only supports json and jsonl files')
    
    return text_data

def tokenize_data(data_path, tokenizer, label_mask_token_id=-100, label_last_n=None, max_length=None, contrastive=False, use_augs=True, mask_neg=False, skip_neg=False):
    '''
    reads in file from data_path and tokenizes it into PRM format
    '''


    if isinstance(data_path, list):
        text_data = []
        for d in data_path:
            text_data.extend(read_json(d))  
    else:
       text_data = read_json(data_path)
 
    tokenize_data = []

    for d in tqdm(text_data):
        tokenize_data.extend(tokenize_one_question(d, 
                                                   tokenizer, 
                                                   label_mask_token_id=label_mask_token_id, 
                                                   label_last_n=label_last_n,
                                                   max_length=max_length,
                                                   contrastive=contrastive,
                                                   use_augs=use_augs,
                                                   mask_neg=mask_neg,
                                                   skip_neg=skip_neg))

    return tokenize_data

class TokenizedPRMDataset(Dataset):
    '''
    Tokenized PRM dataset
    Currently just stores all data in a list

    TODO: do we need to think about better ways to stream in data?
    (Especially for large data)
    '''
    def __init__(self,  
                 data_path, 
                 tokenizer, 
                 label_mask_token_id=-100,
                 label_last_n=None,
                 max_length=None,
                 contrastive=False,
                 use_augs=True,
                 mask_neg=False,
                 skip_neg=False,
                 get_prm800k=False
              ):

        super(TokenizedPRMDataset, self).__init__()

        # hard code using subset of data
            # first get data, then filter out those below 650
        
        self.tokenized_data = tokenize_data(data_path= data_path, 
                                            tokenizer =tokenizer, 
                                            label_mask_token_id=label_mask_token_id, 
                                            label_last_n=label_last_n, 
                                            max_length=max_length, 
                                            contrastive=contrastive,
                                            use_augs=use_augs,
                                            mask_neg=mask_neg,
                                            skip_neg=skip_neg)
        
        if get_prm800k:
            # hardcoded path for now
            prm_data = tokenize_data(data_path= './synth_data/prm800k_train_reprocessed.json', 
                                    tokenizer =tokenizer, 
                                    label_mask_token_id=label_mask_token_id, 
                                    label_last_n=label_last_n, 
                                    max_length=max_length, 
                                    contrastive=contrastive,
                                    use_augs=use_augs,
                                    mask_neg=mask_neg,
                                    skip_neg=skip_neg)
            
            for d in prm_data:
                if len(d.input_ids) > 650: # the max length used for fulltuneing
                    self.tokenized_data.append(d)

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, i):
        return self.tokenized_data[i]
