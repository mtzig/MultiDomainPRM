from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from prm_datasets import TokenizedPRMDataset
import evaluate
import numpy as np
# import torch.nn as nn
from sklearn.metrics import top_k_accuracy_score, roc_auc_score
from scipy.special import softmax

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
    
    t_dataset = TokenizedPRMDataset(configs.train_data_path, 
                                    tokenizer,
                                    label_last_n = configs.train_label_last_n if 'train_label_last_n' in configs else None,
                                    max_length=configs.max_length if 'max_length' in configs else None,
                                    contrastive=configs.contrastive if 'contrastive' in configs else False,
                                    use_augs=configs.use_augs if 'use_augs' in configs else True,
                                    mask_neg= configs.mask_neg if 'mask_neg' in configs else False,
                                    skip_neg= configs.skip_neg if 'skip_neg' in configs else False,
                                    get_prm800k= configs.get_prm800k if 'get_prm800k' in configs else False)
    e_dataset = TokenizedPRMDataset(configs.eval_data_path, 
                                    tokenizer,
                                    label_last_n = configs.eval_label_last_n if 'eval_label_last_n' in configs else None,
                                    max_length=configs.max_length if 'max_length' in configs else None,
                                    contrastive=configs.contrastive if 'contrastive' in configs else False,
                                    use_augs=configs.use_augs if 'use_augs' in configs else True)
    return t_dataset, e_dataset

def get_collate_func(tokenizer):
      
    return DataCollatorForTokenClassification(tokenizer=tokenizer, 
                                                        padding='longest', 
                                                        label_pad_token_id=-100,
                                                        return_tensors='pt')


def get_compute_loss_func():
      
    def compute_loss_func(outputs, labels, num_items_in_batch):
        '''
        '''

        # output logits are in shape (B, L, V) - batch, seq length, vocab size


        logits = outputs.logits[:,:,[12,10]].reshape(-1,2)


        # for eval, num_items_in_batch is None
        if num_items_in_batch is None:
            loss = F.cross_entropy(input=logits,
                            target=labels.flatten(),
                            ignore_index=-100)
            return loss

        loss = F.cross_entropy(input=logits,
                            target=labels.flatten(),
                            ignore_index=-100,
                            reduction='sum')

        return loss / num_items_in_batch
    
    return compute_loss_func


def get_compute_metrics():
    '''
    gets metrics for precision, recall, f1 score

    As for PRM, classifying correctly the wrong reasoning steps is more important,
    we will use wrong reasoing steps as the pos_label
    '''
       
    
    accuracy = evaluate.load('accuracy')
    precision = evaluate.load('precision')
    recall = evaluate.load('recall')
    f1 = evaluate.load('f1')


    def compute_metrics(eval_pred):
        logits, labels = eval_pred


        label_mask_PRM = (labels!=-100)

        labels_PRM = labels[label_mask_PRM]
        logits_PRM = logits[:,:,[12, 10]][label_mask_PRM]

        pred_PRM = np.argmax(logits_PRM, axis=-1)
        predf_PRM = softmax(logits_PRM)[:,1]


        results = {
            'PRM Accuracy': accuracy.compute(predictions=pred_PRM, references=labels_PRM)['accuracy'],
            'PRM Precision': precision.compute(predictions=pred_PRM, references=labels_PRM, zero_division=0.0)['precision'],
            'PRM Recall': recall.compute(predictions=pred_PRM, references=labels_PRM)['recall'],
            'PRM Specificty': recall.compute(predictions=pred_PRM, references=labels_PRM, pos_label=0)['recall'],
            'PRM NPV': precision.compute(predictions=pred_PRM, references=labels_PRM, pos_label= 0, zero_division=0.0)['precision'], # negative predictive value, unPrecision
            'PRM F1': f1.compute(predictions=pred_PRM, references=labels_PRM)['f1'],
            'PRM F1 Neg': f1.compute(predictions=pred_PRM, references=labels_PRM, pos_label=0)['f1'],
            'PRM F1 AUC': roc_auc_score(labels_PRM, pred_PRM),
            'PRM F1 AUC (fixed)': roc_auc_score(labels_PRM, predf_PRM),
            }
    

        return results
    
    return compute_metrics