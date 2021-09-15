import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.crf import CRF
from transformers import BertModel,BertPreTrainedModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy


class Projector:
    def __init__(self, shape_list):
        # shape_list: [var_name]: var_shape, dict， i.e. [name]: [256, 128]
        # project in the input sapce
        self.projectors = {}
        self.alpha_array = np.array([[0.9, 0.001, 0.6]])  # [0.9 * 0.001, 0.6]  # [0.9 * 0.001 ** lamda, 1.0 * 0.1 ** lamda, 0.6]

        for name in shape_list.keys():
            self.projectors[name] = torch.eye(shape_list[name][0]).to(self.args.device)
            # self.parameters[name] = shape_list[name][1]
            # self.alpha_array[name] = shape_list[name][2]
        
    def update(self, projector_data, lambda_=None):

        for data_ in projector_data: #  a sample
            index = 0
            for ind_layer, v in data_: # every layer
                # key_ = self.get_key(name_)
                # pdb.set_trace()
                r = torch.mean(v, 0, keepdim=True)
                projector_ = self.projectors[ind_layer]
                k = torch.matmul(projector_, r.transpose(0, 1))
                dela_P = torch.div(torch.matmul(k, torch.transpose(k, 0, 1)), self.alpha_array[0]*self.alpha_array[1] ** lambda_ + torch.matmul(r, k))
                self.projectors[ind_layer] = torch.sub(self.projectors[ind_layer], dela_P)

    def adjust_gradient(self, model):
        ind = 0
        for n, p in model.named_parameters:
            p.grad = torch.matmul(p.grad, self.projectors[ind])
            ind += 1

    def get_key(self, key):
        for t_ in self.projector_keys:
            if key in t_:
                return t_

        return None