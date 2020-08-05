import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
from otrans.model import Transformer
import  numpy as np
from otrans.metrics import LabelSmoothingLoss
import re

# from    learner import Learner, LearnerTransformer
from    copy import deepcopy

def gen(list1):
    for m in list1:
        yield m

def transferDict(parameters, stateDict):

    parameters = [i for i in parameters]
    keys = [i for i in stateDict]
    # print(keys)
    # print(f"lenParam = {len(parameters)}")
    # print(f"lenKey = {len(keys)}")

    for i, key in enumerate(keys):
        stateDict[key] = parameters[i]

    return stateDict



class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, params):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.device = torch.device('cuda')
        self.crit = LabelSmoothingLoss(size=params['model']['vocab_size'],
                                       smoothing=params['model']['smoothing'])


        self.net = Transformer(params['model']).to(self.device)
        self.net.train()
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, support, query):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        features_spts, features_length_spts, targets_spts, targets_length_spts \
            = support['features_spts'], support['features_length_spts'], support['targets_spts'], support['targets_length_spts']

        features_qrys, features_length_qrys, targets_qrys, targets_length_qrys \
            = query['features_qrys'], query['features_length_qrys'], query['targets_qrys'], query['targets_length_qrys']

        task_num = len(features_spts)
        # print(f"task_num = {task_num}")

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        sum_gradients = []


        for i in range(task_num):

            features_spt = torch.from_numpy(features_spts[i]).to(self.device)

            # print(f"features_spts = {features_spts[i]}")
            # print(f"features_length_spts = {features_length_spts[i]}")
            # print(f"targets_spts = {targets_spts[i]}")
            # print(f"targets_length_spts = {targets_length_spts[i]}")

            # 1. run the i-th task and compute loss for k=0
            features_spt, features_length_spt, targets_spt, targets_length_spt, \
            features_qry, features_length_qry, targets_qry, targets_length_qry\
                = torch.from_numpy(features_spts[i]).to(self.device), torch.from_numpy(features_length_spts[i]).to(self.device), \
                  torch.from_numpy(targets_spts[i]).to(self.device), torch.from_numpy(targets_length_spts[i]).to(self.device), \
                  torch.from_numpy(features_qrys[i]).to(self.device), torch.from_numpy(features_length_qrys[i]).to(self.device), \
                  torch.from_numpy(targets_qrys[i]).to(self.device), torch.from_numpy(targets_length_qrys[i]).to(self.device)


            fast_model = deepcopy(self.net)
            fast_model.to(self.device)
            inner_optimizer = optim.Adam(fast_model.parameters(), lr=self.update_lr)
            fast_model.train()

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = fast_model(features_spt, features_length_spt, targets_spt, targets_length_spt)
                target_out = targets_spt[:, 1:].clone()
                loss = self.crit(logits, target_out)
                # 2. compute grad on theta_pi
                loss.backward()
                # 3. theta_pi = theta_pi - train_lr * grad
                inner_optimizer.step()
                inner_optimizer.zero_grad()

                # print(f"features_qry = {features_qry.size()}")
                logits_q = fast_model(features_qry, features_length_qry, targets_qry, targets_length_qry)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                target_out = targets_qry[:, 1:].clone()
                loss_q = self.crit(logits_q, target_out)
                # print(f"losses_qn = {loss_q}")

                losses_q[k + 1] += loss_q


        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        # print(f"loss_q = {loss_q}")
        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()

        for i, params in enumerate(fast_model.parameters()):
            sum_gradients.append(deepcopy(params.grad))

        del fast_model, inner_optimizer

        for i, params in enumerate(self.net.parameters()):
            params.grad = sum_gradients[i]

        self.meta_optim.step()


        return loss_q


    def finetunning(self, features, features_length, targets, targets_length, fast_model):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """


        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        # net = deepcopy(self.net)
        # print(features)

        logits = fast_model(features, features_length, targets, targets_length)
        # print(f'logits = {logits}')
        target_out = targets[:, 1:].clone()
        loss = self.crit(logits, target_out)



        return loss




def main():
    pass


if __name__ == '__main__':
    main()
