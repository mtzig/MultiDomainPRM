from transformers import Trainer
import torch.nn.functional as F
import torch

PRM = 0
GEN = 1

class JointLossTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        loss computation modified from original hf trainer code to deal with PRM and GEN joint training
        """

        labels = inputs.pop('labels')
        task = inputs.pop('task')
        
        outputs = model(**inputs)

        # we use the last two special tokens for PRM step classification
        # <|reserved_special_token_246|> (ID 128254) for incorrect step
        # <|reserved_special_token_247|> (ID 128255) for correct step


        ## TEMPORARY CHANGE FOR QWEN
        logits_PRM = outputs.logits[task == PRM][:,:,[12,10]].flatten(start_dim=0, end_dim=1)


        # logits_PRM = outputs.logits[task == PRM][:,:,-2:].flatten(start_dim=0, end_dim=1)
        # logits_GEN = outputs.logits[task == GEN][:,:,:-2].flatten(start_dim=0, end_dim=1) # all other tokens are used for next token prediction

        labels_PRM = labels[task == PRM].flatten()
        # labels_GEN = labels[task == GEN]
        # labels_GEN[:,0] = -100 #make sure first token in each row is -100
        # labels_GEN = labels_GEN.flatten()


        # num_items_in_batch is None during eval
        if num_items_in_batch is None:

            loss_PRM = F.cross_entropy(input=logits_PRM,
                            target=labels_PRM,
                            ignore_index=-100) if len(labels_PRM) > 0 else 0
            
            # loss_GEN = F.cross_entropy(input=logits_GEN,
            #                 target=labels_GEN,
            #                 ignore_index=-100) if len(labels_GEN) > 0 else 0

            # TODO We could try some different weightings?
            # loss = loss_PRM + loss_GEN
            loss = loss_PRM


        # during training, num_items_in_batch is not None (and is needed for gradient accumulation)
        else:

            num_items_in_batch_PRM, num_items_in_batch_GEN = num_items_in_batch
            loss_PRM = F.cross_entropy(input=logits_PRM,
                            target=labels_PRM,
                            ignore_index=-100,
                            weight=torch.tensor([20,1],dtype=logits_PRM.dtype, device=logits_PRM.device),
                            reduction='sum') / num_items_in_batch_PRM if len(labels_PRM) > 0 else 0
            
            # loss_GEN = F.cross_entropy(input=logits_GEN,
            #                 target=labels_GEN,
            #                 ignore_index=-100,
            #                 reduction='sum') / num_items_in_batch_GEN if len(labels_GEN) > 0 else 0
            
            # loss = loss_PRM + loss_GEN
            loss = loss_PRM

        
        # if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
        #     loss *= self.accelerator.num_processes

        # return_outputs is only used in prediction step, (not used in our code)
        # but will keep it here, as was in original code
        return (loss, outputs) if return_outputs else loss
    
    def get_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break

        # Keep default behavior the same
        if not self.model_accepts_loss_kwargs:
            return batch_samples, None

        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            # For now we don't support object detection
            try:
                num_items_in_batch_PRM = sum([(batch['labels'][batch['task'] == PRM].ne(-100)).sum() for batch in batch_samples])
                num_items_in_batch_GEN = sum([(batch['labels'][batch['task'] == GEN].ne(-100)).sum() for batch in batch_samples])
                num_items_in_batch = (num_items_in_batch_PRM, num_items_in_batch_GEN)
            except (TypeError, AttributeError):
                pass
        
        # code in original trainer, idk what it does
        # https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L1540
        # if self.args.average_tokens_across_devices:
        #     num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()
        return batch_samples, num_items_in_batch
