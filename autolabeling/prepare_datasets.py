from datasets import load_dataset
from tqdm import tqdm
import random
import json
import os
import yaml
import argparse
from easydict import EasyDict as edict

def get_MC_QA_dataset(dataset='sciq', split='train'):
    '''
        load Multiple Choice QA datasets into list of dictionary of following format

        question
        choices: list
        answer: idx of correct answer
        metadata: dictionary of metadata (dataset dependent)
        id: unique id of form {idx}_{dataset}_{split}
    '''


    data_path = os.path.join(os.path.dirname(__file__), 'datasets',f'{dataset}_{split}.json')

    # if we already downloaded and proccessed data, just directly load it
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            MC_QA_data = json.load(f)

    # otherwise we need to download and process it first
    else:
        MC_QA_data = []

        if dataset=='sciq':
                ds = load_dataset('allenai/sciq')[split]
                
                # we fix a random seed for reproducibility
                random.seed(23420)

                for idx,q in tqdm(enumerate(ds), total=len(ds)):
                    answer_idx = random.randint(0,3)
                    choices = [q['distractor1'],q['distractor2'],q['distractor3']]
                    choices.insert(answer_idx, q['correct_answer'])

                    MC_QA_data.append({'question':q['question'],
                                    'choices': choices,
                                    'answer': answer_idx,
                                    'metadata': {'dataset': dataset,
                                                    'split': split},
                                        'id': f'{idx}_{dataset}_{split}'})
        elif dataset == 'scienceqa':

            ds = load_dataset('derek-thomas/ScienceQA')[split]
            for idx,q in tqdm(enumerate(ds), total= len(ds)):
                
                # skip multimodal questions
                if q['image'] is not None:
                    continue

                MC_QA_data.append({'question':q['question'],
                                'choices': q['choices'],
                                'answer': q['answer'],
                                'metadata': {'dataset': dataset,
                                                'split': split,
                                                'grade': q['grade'],
                                                'topic': q['topic'],
                                                'subject': q['subject']},
                                    'id': f'{idx}_{dataset}_{split}'})
        else:
            raise(NotImplementedError)
        
        # save data so we don't have to process it again in future
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w') as f:
            json.dump(MC_QA_data, f, indent=2)

    return MC_QA_data

def process_MC_QA_question(q):
    '''
    process MC QA question into right format
    e.g.

    Question...?
    A. Choice 1
    B. Choice 2
    C. Choice 3
    ...

    '''
    letters = 'ABCDEFGHIJKLMNOPQ'

    choice_str = ''
    for idx, c in enumerate(q['choices']):
        choice_str += f'\n{letters[idx]}. {c}'
    question = q['question'] + choice_str

    return question


class LLMGen:
    '''
        Object to store all the prompts and parameters we want to use for LLM generation
    '''

    def __init__(self, system_prompt, user_prompt, fewshot_prompt='', id=None, LLM_model='llama', num_gen=1, temp=0.8, max_gen_len=2048):


        self.system_prompt = system_prompt
        self.fewshot_prompt = fewshot_prompt
        self.user_prompt = user_prompt
        self.id = id
        self.LLM_model = LLM_model
        
        if self.LLM_model == 'llama':
            self.LLM_input_prompt = self._convert_prompt_llama()
        else:
            raise(NotImplementedError)



        self.num_gen= num_gen
        self.temp = temp
        self.max_gen_len=max_gen_len

    def _convert_prompt_llama(self):
        '''
        convert prompt to format required by llama
        '''

        # TODO: implement conversion when use fewshot
        if self.fewshot_prompt != '':
            raise(NotImplementedError)
        


        # NOTE: were self.few_shot_prompt is, is where we would want the few shot prompt to be
        converted_prompt = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self.system_prompt}<|eot_id|>{self.fewshot_prompt}<|start_header_id|>user<|end_header_id|>{self.user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

        return converted_prompt
    
    def convert_to_bedrock_input(self):
        '''
        converts to list of dictionaries for bedrock input
        '''


        # for each ID, we add +idx, to ensure they are unique
        return [{'recordID': f'{self.id}+{i}',
                 'modelInput':{'prompt':self.LLM_input_prompt,
                          'temperature':self.temp,
                          'max_gen_len':self.max_gen_len}} for  i in range(self.num_gen)]


def convert_MC_QA_to_LLMGen(MC_QA_data, system_prompt='You\'re a friendly assistant.', fewshot_prompt='', LLM_model='llama', num_gen=1, temp=0.8, max_gen_len=2048):
    '''
    converts MC_QA_data to list of LLMGen objects
    '''
    LLMGens = []
    for q in tqdm(MC_QA_data):
        user_prompt = process_MC_QA_question(q)
        LLMGens.append(LLMGen(system_prompt=system_prompt, 
                              user_prompt=user_prompt, 
                              fewshot_prompt=fewshot_prompt,
                              id=q['id'],
                              LLM_model=LLM_model, 
                              num_gen=num_gen, 
                              temp=temp,
                              max_gen_len=max_gen_len))

    return LLMGens

def convert_LLMGens_to_bedrock_input(LLMGens):
    'convert list of LLMGens to bedrock input'

    bedrock_inputs = []
    for gen in tqdm(LLMGens):
        bedrock_inputs.extend(gen.convert_to_bedrock_input())

    return bedrock_inputs

    




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Script to prepare Bedrock input file.')
    parser.add_argument('-c','--config', type=str, help='Path to config json', default='./gen_cot_configs/sciqqa_trainval_32.yml')
    args = parser.parse_args()

    with open(args.config) as stream:
        try:
            configs = edict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

   


    bedrock_inputs = []
    for data_split in configs.data_splits:
        dataset = data_split.dataset
        for split in data_split.splits:
            MC_QA_data  = get_MC_QA_dataset(dataset=dataset, split=split)

            LLMGens = convert_MC_QA_to_LLMGen(MC_QA_data, 
                                            system_prompt=configs.SYS_PROMPT,
                                            LLM_model=configs.LLM_model,
                                            num_gen=configs.num_gen,
                                            temp=configs.temp,
                                            max_gen_len=configs.max_gen_len)
            
            bedrock_inputs.extend(convert_LLMGens_to_bedrock_input(LLMGens))

    os.makedirs('bedrock_inputs', exist_ok=True)
    with open(f'bedrock_inputs/{configs.experiment_name}.jsonl', 'w') as f:
        for i in bedrock_inputs:
            json.dump(i, f)
            f.write('\n')