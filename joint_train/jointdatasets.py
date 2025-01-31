import json
from tqdm import tqdm
from torch.utils.data import Dataset
from copy import deepcopy

import json
from tqdm import tqdm
from torch.utils.data import Dataset
from copy import deepcopy

PRM = 0
GEN = 1

def merge_dicts(dict_list):
    merged_dict = deepcopy(dict_list[0])
    for d in dict_list[1:]:
        for key, value in d.items():
            merged_dict[key].extend(value)
    return merged_dict

def tokenize_step(cot_step, label, tokenizer, label_mask_token_id=-100, label_last_n=None, label_first_n=None, label_all_correct=False):
    '''
    Specifically tokenizes step for PRM task
    label_last_n: for each step label only the last n as label
    label_first_n: if not none, label the first n of each step as correct
    label_all_correct: if true, labels every token of correct step as correct
    '''
    cot_step_tokenized = tokenizer(cot_step, add_special_tokens=False)


    if label_all_correct and label==1:
        cot_step_labels = [1]* len(cot_step_tokenized.input_ids)
    elif label_last_n is None:
        cot_step_labels = [label]* len(cot_step_tokenized.input_ids)
    else:
        if  label_last_n > len(cot_step_tokenized.input_ids):
            cot_step_labels = [label]*len(cot_step_tokenized.input_ids)
        else:
            cot_step_labels = [label_mask_token_id]*(len(cot_step_tokenized.input_ids)-label_last_n) + [label]*label_last_n

        if label_first_n:

            # only use label_first_n if step is sufficiently large
            if label_first_n <= len(cot_step_labels):
                pass
            cot_step_labels[:label_first_n] = [1]*label_first_n

    
    cot_step_tokenized['labels'] = cot_step_labels

    return cot_step_tokenized


def convert_tokenized_to_gen(tokenized_data, label_mask_length, completed, eos_token_id, label_mask_token_id=-100, eval=False):
    '''
    converts tokenized data to next token prediction format
    completed: true if cot is completed (so we need eos token)
    eval: for evaluation dataset we need to store where whether sample is GEN or PRM using mask token
    '''

    tokenized_data['task'] = GEN

    tokenized_data['labels'] = tokenized_data['input_ids'][1:]

    if completed:
        tokenized_data['labels'].append(eos_token_id)
    else:
        tokenized_data['input_ids'] = tokenized_data['input_ids'][:-1]
        tokenized_data['attention_mask'] = tokenized_data['attention_mask'][:-1]

    # we need to mask out the question (and/or previously trained on tokens), as we don't care about next token prediction on this
    tokenized_data['labels'][:label_mask_length-1] = [label_mask_token_id] * (label_mask_length-1)

    if eval:
        tokenized_data['labels'][0] = -200



def tokenize_one_cot(question_tokenized, data, tokenizer, label_mask_token_id=-100, do_gen=True, label_last_n=None,  label_first_n=None, label_all_correct=False, max_length=None, eval=False, use_augs=True):
    '''
    eval: argument to test eval lose
        Currently not implemented (so does nothing)

    do_gen: true for if we want to do joint training(train as PRM and as generator), otherwise false
    '''

    labels = data['labels']
    
    question_length = len(question_tokenized.input_ids)

    cot_steps_tokenized = []

    # index of first incorrect step (we want tokenization for up to first incorrect step)
    # TODO hacky, can use more elegant implementation later
    first_error_idx = -1

    # true if entire chosen completion is correct else false
    # used for next token prediction (so that we can put in EOS token)
    completed = False

    for i,step in enumerate(data['steps']):
        cot_step = f'\n\n{step}'

        label = 1 if labels[i] == 1 else 0

        cot_step_tokenized = tokenize_step(cot_step, label=label, tokenizer=tokenizer, label_mask_token_id=label_mask_token_id, label_last_n=label_last_n, label_first_n=label_first_n, label_all_correct=label_all_correct)


        cot_steps_tokenized.append(cot_step_tokenized)

        # for incorrect cot, we want to stop after the first incorrect step
        if label == 0:
            first_error_idx = i
            break
        
        # if # Answer is in chosen completion, and we didn't break out of loop due to incorrect step
        # then can assume the entire chosen completion is correct
        if '# Answer' in cot_step:
            completed = True


    augs = []

    # need to keep track of largest aug_idx so that we can figure out if correct cot was completed or not
    max_aug_idx = -1
    if use_augs:
        for aug in data['augs']:
            aug_idx = aug['aug_idx']
            aug_step_content = aug['aug_step']
            aug_step = f'\n\n{aug_step_content}'


            # all augments are incorrect step, except those of type 1 (good) or 0 (okay)
            aug_label = 1 if aug['aug_type'] == 1 or aug['aug_type'] == 0 else 0
            aug_correct = True if aug['aug_type'] == 1 else False # to note if augmentation is correct
            aug_step_tokenized = tokenize_step(aug_step, label=aug_label, tokenizer=tokenizer, label_mask_token_id=label_mask_token_id, label_last_n=label_last_n, label_first_n=label_first_n, label_all_correct=label_all_correct)
            augs.append((aug_step_tokenized, aug_step, aug_idx, aug_correct))

            max_aug_idx = max(max_aug_idx, aug_idx)
 
    tokenized = []

    # chosen_tokenized, is the original full cot that was the chosen completion, from which alternate completions are generated to augment
    chosen_tokenized = merge_dicts([question_tokenized] + cot_steps_tokenized)
    chosen_tokenized['task'] = PRM
    if max_length is None or len(chosen_tokenized.input_ids) <= max_length:
        tokenized.append(chosen_tokenized)


    if first_error_idx != 0 and do_gen:

        gen_chosen_tokenized = deepcopy(chosen_tokenized)
        
        convert_tokenized_to_gen(gen_chosen_tokenized, question_length, 
                                completed = completed, 
                                eos_token_id=tokenizer.eos_token_id,
                                label_mask_token_id=label_mask_token_id,
                                eval=eval)
        
        if max_length is None or len(gen_chosen_tokenized.input_ids) <= max_length:
            tokenized.append(gen_chosen_tokenized)



    # we now change all the labels to masks
    for cot_step_tokenized in cot_steps_tokenized:
        cot_step_tokenized['labels'] = [label_mask_token_id] * len(cot_step_tokenized['labels'])

    for aug in augs:
        aug_step_tokenized, aug_step, aug_idx, aug_correct = aug
        aug_tokenized = merge_dicts([question_tokenized] + cot_steps_tokenized[:aug_idx] + [aug_step_tokenized])
        aug_tokenized['task'] = PRM
        if max_length is None or len(aug_tokenized.input_ids) <= max_length:
            tokenized.append(aug_tokenized)

            if aug_correct and do_gen:
                gen_aug_tokenized = deepcopy(aug_tokenized)

                label_mask_length = len(aug_tokenized.labels) - len(aug_step_tokenized.labels)

                convert_tokenized_to_gen(gen_aug_tokenized, label_mask_length, 
                                completed = '# Answer' in aug_step, 
                                eos_token_id=tokenizer.eos_token_id,
                                label_mask_token_id=label_mask_token_id,
                                eval=eval)
                
                tokenized.append(gen_aug_tokenized)


    return tokenized

def tokenize_one_question(data, tokenizer, label_mask_token_id=-100, do_gen=True, label_last_n=None,  label_first_n=None, label_all_correct=False, max_length=None, eval=False, use_augs=True):
    '''
    can add aug_type param to specify which type of augmentation to use
    '''

    question = data['question']

    question_tokenized = tokenizer(f'{question}\n')

    # we don't want to do token classification on the question and choices part of tokenized
    question_tokenized['labels'] = [label_mask_token_id] * len(question_tokenized.input_ids)
    

    tokenized = []

    for cot in data['chain_of_thoughts']:
        tokenized.extend(tokenize_one_cot(question_tokenized, cot, tokenizer, label_mask_token_id, do_gen, label_last_n, label_first_n, label_all_correct, max_length, eval, use_augs))
    
    return tokenized


def tokenize_data(data_path, tokenizer, label_mask_token_id=-100, do_gen=True, label_last_n=None, label_first_n=None, label_all_correct=False, max_length=None, eval=False, use_augs=True):
    '''
    reads in file from data_path and tokenizes it into PRM format
    '''

    text_data = []

    # support reading a list of json files and combining them
    if isinstance(data_path, list):
        for d in data_path:
            if d.endswith('jsonl'):
                with open(d, 'r') as f:
                    for line in f:
                        text_data.append(json.loads(line))
            elif d.endswith('json'):
                with open(d, 'r') as f:
                    text_data.extend(json.load(f))
            else:
                raise NotImplementedError('currently only supports json and jsonl files')
    else:
        if data_path.endswith('jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    text_data.append(json.loads(line))
        elif data_path.endswith('json'):
            with open(data_path, 'r') as f:
                text_data = json.load(f)
        else:
            raise NotImplementedError('currently only supports json and jsonl files')
    
    tokenize_data = []

    for d in tqdm(text_data):
        tokenize_data.extend(tokenize_one_question(d, 
                                                   tokenizer, 
                                                   label_mask_token_id=label_mask_token_id,
                                                   do_gen=do_gen, 
                                                   label_last_n=label_last_n,
                                                   label_first_n=label_first_n,
                                                   label_all_correct=label_all_correct,
                                                   max_length=max_length,
                                                   eval=eval,
                                                   use_augs=use_augs))

    return tokenize_data


class TokenizedPRMGENDataset(Dataset):
    '''
    Tokenized PRM and Generator dataset
    Currently just stores all data in a list

    TODO: do we need to think about better ways to stream in data?
    (Especially for large data)
    '''
    def __init__(self,  
                 data_path, 
                 tokenizer, 
                 label_mask_token_id=-100,
                 do_gen=True,
                 label_last_n=None,
                 label_first_n=None,
                 label_all_correct=False,
                 max_length=None,
                 eval=False,
                 use_augs=True
              ):

        super(TokenizedPRMGENDataset, self).__init__()
        
        self.tokenized_data = tokenize_data(data_path= data_path, 
                                            tokenizer =tokenizer, 
                                            label_mask_token_id=label_mask_token_id,
                                            do_gen=do_gen, 
                                            label_last_n=label_last_n, 
                                            label_first_n=label_first_n,
                                            label_all_correct=label_all_correct,
                                            max_length=max_length, 
                                            eval=eval,
                                            use_augs=use_augs)

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, i):
        return self.tokenized_data[i]
