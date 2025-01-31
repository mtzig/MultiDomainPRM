from dataset_utils import get_qa_dataset
from tqdm import tqdm
import json
import os
import yaml
import argparse
from easydict import EasyDict as edict


def create_llama_fewshot(fs):
    fewshot_str = ''
    for f in fs:
        q = f['question']
        exp = f['exp']

        fewshot_str += f'<|start_header_id|>user<|end_header_id|>{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{exp}<|eot_id|>'

    return fewshot_str

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
        # if self.fewshot_prompt != '':
        #     raise(NotImplementedError)
        


        # NOTE: where self.few_shot_prompt is, is where we would want the few shot prompt to be
        converted_prompt = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self.system_prompt}<|eot_id|>{self.fewshot_prompt}<|start_header_id|>user<|end_header_id|>{self.user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

        return converted_prompt
    
    def convert_to_bedrock_input(self):
        '''
        converts to list of dictionaries for bedrock input
        '''

        if self.LLM_model == 'llama':
            # for each ID, we add +idx, to ensure they are unique
            return [{'recordId': f'{self.id}+{i}',
                    'modelInput':{'prompt':self.LLM_input_prompt,
                            'temperature':self.temp,
                          'max_gen_len':self.max_gen_len}} for  i in range(self.num_gen)]
        
        else:
            raise(NotImplementedError)


def convert_qa_to_LLMGens(qa_data, system_prompt='You\'re a friendly assistant.', fewshot_prompt='', LLM_model='llama', num_gen=1, temp=0.8, max_gen_len=2048):
    '''
    converts qa_data to list of LLMGen objects
    '''
    LLMGens = []
    for q in tqdm(qa_data):
        user_prompt = q['question']
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

    fewshot_prompt = ''
    if 'fewshot_path' in configs:
        with open(configs.fewshot_path, 'r') as f:
            fewshot_ds = json.load(f)

        fewshot_prompt = create_llama_fewshot(fewshot_ds)
 

    bedrock_inputs = []
    for data_split in configs.data_splits:
        dataset = data_split.dataset
        subsample = data_split.subsample if 'subsample' in data_split else None
        subsample_seed = data_split.subsample_seed if 'subsample_seed' in data_split else None

        for split in data_split.splits:

            

            qa_data  = get_qa_dataset(dataset=dataset, split=split, subsample=subsample, subsample_seed=subsample_seed)

            LLMGens = convert_qa_to_LLMGens(qa_data, 
                                            system_prompt=configs.SYS_PROMPT,
                                            fewshot_prompt=fewshot_prompt,
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