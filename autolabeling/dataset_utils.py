from datasets import load_dataset
from tqdm import tqdm
import random
import json
import os
from bs4 import BeautifulSoup
import re
from collections import defaultdict

def strip_html(html_content):
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_content, 'html.parser')
    # Extract and return plain text
    return soup.get_text()

def process_mc_qa(question, choices, answer_idx):
    '''
    question: the question string
    choices: list of the answer choices
    answer_idx: idx of the correct answer in choices

    returns question_processed, answer

    process multiple choice question into right format
    e.g.

    Question...?
    A. Choice 1
    B. Choice 2
    C. Choice 3

    ...

    '''
    letters = 'ABCDEFGHIJKLMNOPQ'

    choice_str = ''
    for idx, c in enumerate(choices):
        choice_str += f'\n{letters[idx]}. {c}'

    question_processed = question + choice_str

    answer = letters[answer_idx]

    return question_processed, answer

def get_qa_dataset(dataset='sciq', split='train', subsample=None, subsample_seed=None):
    '''
        load QA datasets into list of dictionary of following format

        question: question string
        answer: correct answer
        metadata: dictionary of metadata (dataset dependent)
        id: unique id of form {idx}_{dataset}_{split}
    '''

    module_path = os.path.dirname(__file__)

    data_path = os.path.join(module_path, 'datasets',f'{dataset}_{split}.json')

    # if we already downloaded and proccessed data, just directly load it
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            qa_data = json.load(f)

    # otherwise we need to download and process it first
    else:
        qa_data = []

        if dataset=='sciq':
                ds = load_dataset('allenai/sciq')[split]
                
                # we fix a random seed for reproducibility
                random.seed(23420)

                for idx,q in tqdm(enumerate(ds), total=len(ds)):
                    answer_idx = random.randint(0,3)
                    choices = [q['distractor1'],q['distractor2'],q['distractor3']]
                    choices.insert(answer_idx, q['correct_answer'])

                    question, answer = process_mc_qa(q['question'],
                                                      choices,
                                                      answer_idx)

                    qa_data.append({'question': question,
                                    'answer': answer,
                                    'metadata': {'dataset': dataset,
                                                    'split': split},
                                        'id': f'{idx}_{dataset}_{split}'})
        elif dataset == 'scienceqa':

            ds = load_dataset('derek-thomas/ScienceQA')[split]
            for idx,q in tqdm(enumerate(ds), total= len(ds)):
                
                # skip multimodal questions
                if q['image'] is not None:
                    continue
                
                question, answer = process_mc_qa(q['question'],
                                    q['choices'],
                                    q['answer'])

                qa_data.append({'question': question,
                                'answer': answer,
                                'metadata': {'dataset': dataset,
                                                'split': split,
                                                'grade': q['grade'],
                                                'topic': q['topic'],
                                                'subject': q['subject']},
                                    'id': f'{idx}_{dataset}_{split}'})
        elif dataset == 'arlsat':
            ds = load_dataset('tasksource/lsat-ar')[split]
            for idx,q in tqdm(enumerate(ds), total= len(ds)):
                
                context = q['context']
                question = q['question']
                question, answer = process_mc_qa(f'{context}\n Question: {question}',
                                    q['answers'],
                                    q['label'])

                qa_data.append({'question': question,
                                'answer': answer,
                                'metadata': {'dataset': dataset,
                                                'split': split,
                                                'orig_id': q['id_string']},
                                    'id': f'{idx}_{dataset}_{split}'})

        elif dataset == 'chemqa':
            ds = load_dataset('avaliev/ChemistryQA')[split]
            for idx,q in tqdm(enumerate(ds), total= len(ds)):
                
                # we only want physical unit questions
                if 'physical unit' not in q['answer_json']:
                    continue
                
                target_var = eval(q['target_var_json'])[0]['value'].replace('[','').replace(']','')

                question = q['question'] + f' Your answer should be in the form {target_var}.'
                answer = q['answer']

                qa_data.append({'question': question,
                                'answer': answer,
                                'explanation': re.sub(r'^[\s\n]*Explanation:[\s\n]*', '', 
                                                      strip_html(q['answer_description'])),
                                'metadata': {'dataset': dataset,
                                                'split': split,
                                                'target_var_json': q['target_var_json'],
                                                'guid': q['guid'],
                                                'url': q['url']},
                                'id': f'{idx}_{dataset}_{split}'})
        elif dataset == 'formular':
            
            raw_data_path = os.path.join(module_path, 'datasets/FormulaReasoning/en_FormulaReasoning.json')

            if split == 'train':
                split_path = os.path.join(module_path, 'datasets/FormulaReasoning/train.json')
            elif split == 'test':
                split_path = os.path.join(module_path, 'datasets/FormulaReasoning/id_test.json')
            else:
                raise(KeyError('formular only supports train and test splits'))
            
            with open(split_path, 'r') as f:
                split_data = json.load(f)

            ids = set([d['id'] for d in split_data])

            with open(raw_data_path, 'r') as f:
                ds = json.load(f)

            for idx,q in tqdm(enumerate(ds), total= len(ds)):
                
                id = q['id']

                # only want question of the specified split
                if id not in ids:
                    continue

                question = q['question'].strip()
                answer = q['answer']

                qa_data.append({'question': question,
                                'answer': answer,
                                'explanation':  re.sub(r'^[\s\n]*Explanation:[\s\n]*', '', q['standard_explanation']),
                                'metadata': {'dataset': dataset,
                                                'split': split,
                                                'orig_id': q['id']},
                                'id': f'{idx}_{dataset}_{split}'})
        elif dataset == 'medmcqa':

            if split != 'train':
                raise(NotImplementedError) # current splitting only for train
            
            random.seed(9013424) # fixed seed for reproducibility

            subj_count = defaultdict(int)

            ds = load_dataset("openlifescienceai/medmcqa")[split]
            for idx,q in tqdm(enumerate(ds), total= len(ds)):
                subj = q['subject_name']
                exp = q['exp']

                if exp is None or len(exp) < 150: # exclude questions with short explanations
                    continue
                elif subj == 'Medicine' or subj == 'Psychiatry': 
                    # we want a custom balanced subset of train data
                    if subj_count[subj] >= 1000:
                        continue
                    else:
                        subj_count[subj] += 1
                elif subj == 'Biochemistry' or subj == 'Microbiology':
                    if subj_count[subj] >= 2000:
                        continue
                    else:
                        subj_count[subj] += 1
                else:
                    if subj_count['other'] >= 2000:
                        continue
                    else:
                        subj_count['other'] += 1

                answer_idx = q['cop']
                choices = [q['opa'],q['opb'],q['opc'], q['opd']]

                question, answer = process_mc_qa(q['question'],
                                                 choices,
                                                 answer_idx)

                qa_data.append({'question': question,
                                'answer': answer,
                                'explanation':  exp,
                                'metadata': {'dataset': dataset,
                                                'split': split,
                                                'subject':subj,
                                                'topic': q['topic_name'],
                                                'orig_id': q['id']},
                                'id': f'{idx}_{dataset}_{split}'})
        else:
            raise(NotImplementedError)
        
        # save data so we don't have to process it again in future
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w') as f:
            json.dump(qa_data, f, indent=2)

    # for getting a random subsampled subset
    if subsample != None:
        random.seed(subsample_seed)
        random.shuffle(qa_data)
        qa_data = qa_data[:subsample]

    return qa_data

