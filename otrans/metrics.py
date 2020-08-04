import torch
from torch import nn
import torch.nn.functional as F
from otrans.data import PAD


from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, smoothing, alpha=None, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.smoothing = smoothing
        self.confidence = 1 - smoothing

    def forward(self, inputs, targets):
        N = inputs.size(0)  # inputs是网络的top层的输出
        C = inputs.size(1)

        # maxinputs = torch.max(inputs,1)[0]
        # inputs = inputs - maxinputs.view(-1,1)
        P = F.softmax(inputs)  # 先求p_t

        # print(f"inputs2 = {inputs}")
        # print(inputs.size())
        # print(f"targets = {targets}")
        # print(f"P = {P}")


        class_mask = inputs.data.new(N, C).fill_(self.smoothing / (self.class_num - 1))
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, self.confidence)  # 得到label的one_hot编码
        # print(f"class_mask = {class_mask}")

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        # print(f"alpha = {alpha.size()}")
        # y*p_t  如果这里不用*， 还可以用gather提取出正确分到的类别概率。
        # 之所以能用sum，是因为class_mask已经把预测错误的概率清零了。
        probs = (P * class_mask).sum(1).view(-1, 1)
        # print(f"probs = {probs}")
        # y*log(p_t)
        # log_p = probs.log()
        log_p = (torch.log_softmax(inputs, dim=1) * class_mask).sum(1).view(-1, 1)
        # print(f"log_p = {log_p}")
        # -a * (1-p_t)^2 * log(p_t)
        # print(f"1 - probs = {1 - probs}")
        # print(f"torch.pow = {torch.pow((1 - probs), self.gamma)}")
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        # print(loss)
        return loss

class FocalLossPack(nn.Module):
    """Label-smoothing loss

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(self, size, smoothing):
        super(FocalLossPack, self).__init__()
        self.criterion = FocalLoss(class_num=size, smoothing=smoothing, gamma=2, size_average=True)
        # self.criterion = nn.CrossEntropyLoss()
        self.size = size


    def forward(self, x, target):
        """Compute loss between x and target

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target: target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        # print(f"x = {x}")
        # print(f"x_size = {x.size()}")
        # print(f"target = {target}")
        # print(f"target_size = {target.size()}")
        x = x.view(-1, self.size)
        target = target.reshape(-1)


        Fl = self.criterion(x, target)

        return Fl

class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(self, size, smoothing, padding_idx=PAD, normalize_length=True,
                 criterion=nn.KLDivLoss(reduction='none')):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target: target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.reshape(-1)
        # print(f"target = {target}")
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            # print(f"self.padding_idx = {self.padding_idx}")
            # print(f"ignore = {ignore}")
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
