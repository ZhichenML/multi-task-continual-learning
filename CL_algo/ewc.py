import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
from torch.utils.data import DataLoader
import gc


class EWC_LOSS:
    # https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks
    def __init__(self, model):
        self.model = model
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _compute_consolidation_loss(self, weight=1000000):
            try:
                losses = []
                for param_name, param in self.model.named_parameters():
                    _buff_param_name = param_name.replace('.', '__')
                    estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                    estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                    losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
                return (weight / 2) * sum(losses)
            except AttributeError:
                return 0
                
    def _update_fisher_params(self, dl, num_batch, num_labels):
            # dl = DataLoader(current_ds, batch_size, shuffle=True)
            log_liklihoods_ner = []
            log_liklihoods_cates = []
            
            for i, batch in enumerate(dl):
                print("batch size", len(batch))
                if i > num_batch:
                    break
                batch = tuple(t.to(torch.device("cuda")) for t in batch)
  
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2], "labels": batch[3],'cates_ids': batch[5],'acts_ids': batch[6]}
                print(i)
                outputs, loss_cates, logits_cates = self.model(**inputs)
                logits_ner = outputs[1]
                
                attention_mask = inputs["attention_mask"]
                active_loss = attention_mask.view(-1) == 1 #contiguous()
                
                active_logits_ner = logits_ner.view(-1, num_labels)[active_loss]

                output_ner = F.log_softmax(active_logits_ner, dim=1)
                active_labels = inputs["labels"].view(-1)[active_loss]

                log_liklihoods_ner.append(torch.gather(output_ner, dim=1, index=active_labels.unsqueeze(-1)) )
        
                output_cates = F.log_softmax(logits_cates, dim=1)
                log_liklihoods_cates.append(torch.gather(output_cates, dim=1, index=inputs["cates_ids"].unsqueeze(-1)))
                

                log_likelihood_ner = torch.cat(log_liklihoods_ner).mean()
            
                log_likelihood_cates = torch.cat(log_liklihoods_cates).mean()
                grad_log_liklihood1 = autograd.grad(log_likelihood_ner, self.model.parameters(), allow_unused=True, retain_graph=True) #+ \
                grad_log_liklihood1 = [torch.tensor(0) if v == None else v for v in grad_log_liklihood1]
                self.model.zero_grad()
                grad_log_liklihood2 = autograd.grad(log_likelihood_cates, self.model.parameters(), allow_unused=True, retain_graph=True)
                grad_log_liklihood2 = [torch.tensor(0) if v == None else v for v in grad_log_liklihood2]
                
                grad_log_liklihood = grad_log_liklihood1 + grad_log_liklihood2

        
                _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
                for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
                    
                    if hasattr(self.model, '{}_estimated_fisher'.format(_buff_param_name)):
                        estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                        self.model.register_buffer(_buff_param_name+'_estimated_fisher', estimated_fisher+param.data.clone() ** 2)
                    else:
                        self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

                gc.collect()

                torch.cuda.empty_cache()


    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def register_ewc_params(self, dl, num_batches, num_labels):
        self._update_fisher_params(dl, num_batches, num_labels)
        self._update_mean_params()


class EWC_LOSS_span:
    # https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks
    def __init__(self, model):
        self.model = model
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _compute_consolidation_loss(self, weight=1000000):
            try:
                losses = []
                for param_name, param in self.model.named_parameters():
                    _buff_param_name = param_name.replace('.', '__')
                    estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                    estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                    losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
                return (weight / 2) * sum(losses)
            except AttributeError:
                return 0
                
    def _update_fisher_params(self, dl, num_batch, num_labels):
            # dl = DataLoader(current_ds, batch_size, shuffle=True)
            log_liklihoods_ner_start = []
            log_liklihoods_ner_end = []
            log_liklihoods_cates = []
            for i, batch in enumerate(dl):
                if i > num_batch:
                    break
                batch = tuple(t.to(torch.device("cuda")) for t in batch)
  
                # 1. get logits
                inputs = {"input_ids": batch[0], "token_type_ids":batch[2] , "attention_mask": batch[1],
                        "start_positions": batch[3], "end_positions": batch[4],'cates_ids': batch[6],'acts_ids': batch[7]}
                        
                outputs, loss_cates, logits_cates = self.model(**inputs)
                start_logits = outputs[1]
                end_logits = outputs[2]

                attention_mask = inputs["attention_mask"]
                active_loss = attention_mask.view(-1) == 1

               
                active_start_logits = start_logits.view(-1, num_labels)[active_loss]
                active_end_logits = end_logits.view(-1, num_labels)[active_loss]

                active_start_labels = inputs["start_positions"].view(-1)[active_loss]
                active_end_labels = inputs["end_positions"].view(-1)[active_loss]

                
                # 2. get log likelihood
                output_start = F.log_softmax(active_start_logits, dim=1)
                log_liklihoods_ner_start.append(torch.gather(output_start, dim=1, index=active_start_labels.unsqueeze(-1)) )
                
                
                output_end = F.log_softmax(active_end_logits, dim=1)
                log_liklihoods_ner_end.append(torch.gather(output_end, dim=1, index=active_end_labels.unsqueeze(-1)) )
                

                output_cates = F.log_softmax(logits_cates, dim=1)
                log_liklihoods_cates.append(torch.gather(output_cates, dim=1, index=inputs["cates_ids"].unsqueeze(-1)) )

            # 3. mean likelihood and grad
            log_likelihood_ner_start = torch.cat(log_liklihoods_ner_start).mean()
            log_likelihood_ner_end = torch.cat(log_liklihoods_ner_end).mean()
            log_likelihood_cates = torch.cat(log_liklihoods_cates).mean()

            grad_log_liklihood_ner_start = autograd.grad(log_likelihood_ner_start, self.model.parameters(), allow_unused=True, retain_graph=True)
            grad_log_liklihood_ner_start = [torch.tensor(0) if v == None else v for v in grad_log_liklihood_ner_start]
            grad_log_liklihood_ner_end = autograd.grad(log_likelihood_ner_end, self.model.parameters(), allow_unused=True, retain_graph=True)
            grad_log_liklihood_ner_end = [torch.tensor(0) if v == None else v for v in grad_log_liklihood_ner_end]
            grad_log_liklihood_cates = autograd.grad(log_likelihood_cates, self.model.parameters(), allow_unused=True)
            grad_log_liklihood_cates = [torch.tensor(0) if v == None else v for v in grad_log_liklihood_cates]
            grad_log_liklihood = grad_log_liklihood_ner_start + grad_log_liklihood_ner_end + grad_log_liklihood_cates


            _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
            for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
                self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def register_ewc_params(self, dl, num_batches, num_labels):
        self._update_fisher_params(dl, num_batches, num_labels)
        self._update_mean_params()




class ElasticWeightConsolidation:

    def __init__(self, model, crit, lr=0.001, weight=1000000):
        self.model = model
        self.weight = weight
        self.crit = crit
        self.optimizer = optim.Adam(self.model.parameters(), lr)

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def _update_fisher_params(self, dl, num_batch):
        # dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_liklihoods = []
        for i, (input, target) in enumerate(dl):
            if i > num_batch:
                break
            output = F.log_softmax(self.model(input), dim=1)
            log_liklihoods.append(output[:, target])
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def register_ewc_params(self, dataset, batch_size, num_batches):
        self._update_fisher_params(dataset, batch_size, num_batches)
        self._update_mean_params()

    def _compute_consolidation_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0


    



    def forward_backward_update(self, input, target):
        output = self.model(input)
        loss = self._compute_consolidation_loss(self.weight) + self.crit(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)




