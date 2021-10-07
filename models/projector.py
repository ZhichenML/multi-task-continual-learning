import pdb

import torch


class Projector:
    def __init__(self, args):
        self.is_inited = False
        self.args = args
        self.projectors = {}
        self.parameters = {}
        self.projector_keys = None
        self.alpha_array = {}  # [0.9 * 0.001, 0.6]  # [0.9 * 0.001 ** lamda, 1.0 * 0.1 ** lamda, 0.6]

    def initial(self, shape_list):
        self.is_inited = True

        for name_ in shape_list:
            self.projectors[name_] = torch.eye(shape_list[name_][0][1]).to(self.args.device)
            self.parameters[name_] = shape_list[name_][1]
            self.alpha_array[name_] = shape_list[name_][2]
        # pdb.set_trace()
        self.projector_keys = tuple(self.projectors.keys())

    def update(self, projector_data, lambda_):

        for data_ in projector_data:
            index = 0
            for name_ in data_:
                key_ = self.get_key(name_)
                # pdb.set_trace()
                r = torch.mean(data_[name_], 0, keepdim=True)
                projector_ = self.projectors[key_]
                k = torch.matmul(projector_, r.transpose(0, 1))
                dela_P = torch.div(torch.matmul(k, torch.transpose(k, 0, 1)), self.alpha_array[key_][index] ** lambda_ + torch.matmul(r, k))
                self.projectors[key_] = torch.sub(self.projectors[key_], dela_P)

    def adjust_gradient(self):
        for name_ in self.parameters:
            self.parameters[name_].grad = torch.matmul(self.parameters[name_].grad, self.projectors[name_])

    def get_key(self, key):
        for t_ in self.projector_keys:
            if key in t_:
                return t_

        return None
