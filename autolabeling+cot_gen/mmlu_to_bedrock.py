from collections import defaultdict
import json
from datasets import load_dataset
import random
from tqdm import tqdm
import re
import os

def get_template(data):
    '''
    data is list of tuples, each tuple is question and corresponding CoT
    '''

    q_template = 'Given the following question and candidate answers, choose the best answer.\n{question}\n\nYour response should end with "The answer is (X)." where X is a letter from the provided choices.\nEach reasoning step in your response should be delimited by two newline characters\n\nLet\'s think step by step.'

    template = ''
    for q, cot in data:
        q_prompt = q_template.format(question=q)
        template += f'<|start_header_id|>user<|end_header_id|>\n\n{q_prompt}<|eot_id|>'
        template += f'<|start_header_id|>assistant<|end_header_id|>\n\n{cot}<|eot_id|>'

    template += f'<|start_header_id|>user<|end_header_id|>\n\n{q_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

    return template

def process_question(data):
    letters = 'ABCDEFGHIJKLMNOP'
    choice_str = ''
    for idx, c in enumerate(data['options']):
        choice_str += f'\n{letters[idx]}. {c}'

    question = 'Question: ' + data['question'] + choice_str
    return question

def get_mmlu_pro_fewshot_templates(data_path='mmlu_val.json'):
    with open(data_path, 'r') as f:
        mmlu_data = json.load(f)

    # key will be the category, value will be list of question, cot pairs
    fewshot_dict = defaultdict(list)
    
    for d in mmlu_data:

        fewshot_dict[d['category']].append((process_question(d), d['cot_content']))

    for subject in fewshot_dict:
        fewshot_dict[subject] = get_template(fewshot_dict[subject])

    return fewshot_dict

def get_mmlu(samples=100, after_idx=0):

    random.seed(9234232) # fix seed for reproducibility
    ds = list(load_dataset("TIGER-Lab/MMLU-Pro")['test'])
    random.shuffle(ds)

    # store count of questions in each subject
    mmlu_questions = defaultdict(int)
    mmlu_after = defaultdict(int)

    qa_data = []

    for d in tqdm(ds):
        category = d['category']


        if mmlu_after[category] < after_idx:
            # weired thing for when we want to ignore first number of questions as use that for testing
            mmlu_after[category] += 1
            continue
        if mmlu_questions[category] >= samples:
            continue
        else:
            mmlu_questions[category] += 1

        id = d['question_id']

        qa_data.append({'question': process_question(d),
                'answer': d['answer'],
                'metadata': {'category': category,
                                'src': d['src']},
                'id': f'{id}_mmlu_{category}'})
        
    return qa_data

def add_llm_prompts(qa_data):

    templates = get_mmlu_pro_fewshot_templates()
    for qa in qa_data:
        template = templates[qa['metadata']['category']]
        # print(qa['metadata']['category'])
        # print(qa['question'])

        # very hacky way to add in question
        qa['llm_prompt'] = template[:-274] + qa['question'] + template[-264:]

        #re.sub(r'\{question\}', re.escape(qa['question']), template)



def convert_mmlu_to_bedrock_input(qa, temp=0.8, max_gen_len=2-48, num_gen=128):
    '''
    converts to list of dictionaries for bedrock input
    '''



    # for each ID, we add +idx, to ensure they are unique
    return [{'recordId': f'{qa['id']}+{i}',
                'modelInput':{'prompt':qa['llm_prompt'],
                        'temperature':temp,
                        'max_gen_len':max_gen_len}} for  i in range(num_gen)]


if __name__=='__main__':

    samples = 500
    num_gen = 16
    after_idx = 150 # or 0

    temp = 0.8
    max_gen_len= 2048

    qa_data = get_mmlu(samples=samples, after_idx=after_idx)
    add_llm_prompts(qa_data)

    # make add_llm_prompts v2 

    bedrock_inputs = []
    for qa in qa_data:
        bedrock_inputs.extend(convert_mmlu_to_bedrock_input(qa,
                                                            temp=temp,
                                                            max_gen_len=max_gen_len,
                                                            num_gen=num_gen))

    os.makedirs('bedrock_inputs', exist_ok=True)
    with open(f'bedrock_inputs/mmlu_{samples}_{num_gen}.jsonl', 'w') as f:
        for i in bedrock_inputs:
            json.dump(i, f)
            f.write('\n')