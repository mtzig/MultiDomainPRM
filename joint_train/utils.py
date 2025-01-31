from transformers import AutoTokenizer,AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from jointdatasets import TokenizedPRMGENDataset
import evaluate
import numpy as np
# import torch.nn as nn
import torch
from sklearn.metrics import top_k_accuracy_score, roc_auc_score

VOCAB_SIZE = 128256 # Vocab size of Llama


def get_model(configs):
    '''
    right now just returns huggingface model
    might be useful to have this method, if we want to do more complicated stuff
    '''
    

    # model = AutoModelForTokenClassification.from_pretrained(configs.model_id)

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)


    if 'lora_config' in configs:
        print('Using LoRA')
        lora_config = LoraConfig(**configs.lora_config)
        model = get_peft_model(model, lora_config)
        
    return model

def get_tokenizer(model_id):
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token #llama doesn't define pad token, so we need to do this
    tokenizer.padding_side='right' # we need to pad from right (so that we can do eval mask id trick for eval)

    global VOCAB_SIZE
    VOCAB_SIZE = len(tokenizer)

    return tokenizer

def get_datasets(configs, tokenizer):
    
    t_dataset = TokenizedPRMGENDataset(configs.train_data_path, 
                                    tokenizer,
                                    do_gen = configs.do_gen if 'do_gen' in configs else True,
                                    label_last_n = configs.train_label_last_n if 'train_label_last_n' in configs else None,
                                    label_first_n = configs.train_label_first_n if 'train_label_first_n' in configs else None,
                                    label_all_correct = configs.train_label_all_correct if 'train_label_all_correct' in configs else False,
                                    max_length=configs.max_length if 'max_length' in configs else None,
                                    eval=False,
                                    use_augs=configs.use_augs if 'use_augs' in configs else True)
    e_dataset = TokenizedPRMGENDataset(configs.eval_data_path, 
                                    tokenizer,
                                    do_gen = configs.do_gen if 'do_gen' in configs else True,
                                    label_last_n = configs.eval_label_last_n if 'eval_label_last_n' in configs else None,
                                    max_length=configs.max_length if 'max_length' in configs else None,
                                    eval=True,
                                    use_augs=configs.use_augs if 'use_augs' in configs else True)
    return t_dataset, e_dataset

def get_collate_func(tokenizer):
      
    return DataCollatorForTokenClassification(tokenizer=tokenizer, 
                                                        padding='longest', 
                                                        label_pad_token_id=-100,
                                                        return_tensors='pt')


def get_compute_metrics(do_gen=True, model_type='llama'):
    '''
    gets metrics for precision, recall, f1 score

    As for PRM, classifying correctly the wrong reasoning steps is more important,
    we will use wrong reasoing steps as the pos_label
    '''
       
    
    accuracy = evaluate.load('accuracy')
    precision = evaluate.load('precision')
    recall = evaluate.load('recall')
    f1 = evaluate.load('f1')
    loss = evaluate.load('mtzig/cross_entropy_loss')

    if do_gen:
        def compute_metrics(eval_pred):
            logits, labels = eval_pred

            PRM_idx = labels[:,0] == -100
            GEN_idx = ~PRM_idx

            # after using labels to extract idx info, need to restore to mask id
            labels[:,0] = -100
            
            # redo this in for loop?
            labels_PRM = labels[PRM_idx]
            label_mask_PRM = (labels_PRM!=-100)

            labels_PRM = labels_PRM[label_mask_PRM]

            logits_PRM = logits[PRM_idx][:,:,-2:][label_mask_PRM]
            pred_PRM = np.argmax(logits_PRM, axis=-1)



            labels_GEN = labels[GEN_idx]
            label_mask_GEN = (labels_GEN!=-100)
            labels_GEN = labels_GEN[label_mask_GEN]
            logits_GEN = logits[GEN_idx][:,:,:-2][label_mask_GEN]

            results = {
                'PRM Accuracy': accuracy.compute(predictions=pred_PRM, references=labels_PRM)['accuracy'],
                'PRM Precision': precision.compute(predictions=pred_PRM, references=labels_PRM, zero_division=0.0)['precision'],
                'PRM Recall': recall.compute(predictions=pred_PRM, references=labels_PRM)['recall'],
                'PRM Specificty': recall.compute(predictions=pred_PRM, references=labels_PRM, pos_label=0)['recall'],
                'PRM NPV': precision.compute(predictions=pred_PRM, references=labels_PRM, pos_label= 0, zero_division=0.0)['precision'], # negative predictive value, unPrecision
                'PRM F1': f1.compute(predictions=pred_PRM, references=labels_PRM)['f1'],
                'PRM F1 Neg': f1.compute(predictions=pred_PRM, references=labels_PRM, pos_label=0)['f1'],
                'PRM F1 AUC': roc_auc_score(labels_PRM, pred_PRM),
                'PRM Loss': F.cross_entropy(input=torch.from_numpy(logits_PRM),
                                target=torch.from_numpy(labels_PRM),
                                ignore_index=-100),
                'GEN Loss': F.cross_entropy(input=torch.from_numpy(logits_GEN),
                                target=torch.from_numpy(labels_GEN),
                                ignore_index=-100),
                'GEN top-5 accuracy': top_k_accuracy_score(labels_GEN, logits_GEN, k=5, labels=np.arange(VOCAB_SIZE-2)).item() # hardcode vocab size to that of llama 3.1 (minus the last two special tokens)
            }
        

            return results
    else:

        def compute_metrics(eval_pred):
            logits, labels = eval_pred

            PRM_idx = labels[:,0] == -100


            # after using labels to extract idx info, need to restore to mask id

            
            # redo this in for loop?
            labels_PRM = labels[PRM_idx]
            label_mask_PRM = (labels_PRM!=-100)

            labels_PRM = labels_PRM[label_mask_PRM]

            if model_type == 'qwen2':
                # 12 is ID for -
                # 10 is ID for +
                logits_PRM = logits[PRM_idx][:,:,[12, 10]][label_mask_PRM]
            else:
                logits_PRM = logits[PRM_idx][:,:,-2:][label_mask_PRM]

            pred_PRM = np.argmax(logits_PRM, axis=-1)


            results = {
                'PRM Accuracy': accuracy.compute(predictions=pred_PRM, references=labels_PRM)['accuracy'],
                'PRM Precision': precision.compute(predictions=pred_PRM, references=labels_PRM, zero_division=0.0)['precision'],
                'PRM Recall': recall.compute(predictions=pred_PRM, references=labels_PRM)['recall'],
                'PRM Specificty': recall.compute(predictions=pred_PRM, references=labels_PRM, pos_label=0)['recall'],
                'PRM NPV': precision.compute(predictions=pred_PRM, references=labels_PRM, pos_label= 0, zero_division=0.0)['precision'], # negative predictive value, unPrecision
                'PRM F1': f1.compute(predictions=pred_PRM, references=labels_PRM)['f1'],
                'PRM F1 Neg': f1.compute(predictions=pred_PRM, references=labels_PRM, pos_label=0)['f1'],
                'PRM F1 AUC': roc_auc_score(labels_PRM, pred_PRM),
                }
        

            return results
    
    return compute_metrics