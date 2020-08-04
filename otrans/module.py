import math
import torch
# import dgl
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple


class G(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(G, self).__init__()
        self.G = self._build_karate_club_graph(vocab_size)
        print('We have %d nodes.' % self.G.number_of_nodes())
        print('We have %d edges.' % self.G.number_of_edges())
        self.G.ndata['feat'] = nn.Embedding(vocab_size, d_model).weight
        self.G.ndata['feat'] = self.G.ndata['feat'].cuda()
    def _build_karate_club_graph(self, size):
        # All 78 edges are stored in two numpy arrays. One for source endpoints
        # while the other for destination endpoints.
        G = dgl.DGLGraph()
        G.add_nodes(size)

        list1 = [x for x in range(size)]
        list2 = torch.Tensor(list1).long()

        for i in list1:
            G.add_edges(list2, i)

        # Construct a DGLGraph
        return G
    def forward(self,x):
        return self.G.ndata['feat'][x]


class LearnedSwish(nn.Module):
    def __init__(self,slope=1):
        super().__init__()
        self.slope = (slope * torch.nn.Parameter(torch.ones(1))).cuda()

    def forward(self,x):
        return x * torch.sigmoid(self.slope*x)



class Embeddings(nn.Embedding):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__(vocab, d_model)
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class.

        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length

        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * 
                             -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]

        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.

    See also: Sec. 3.2  https://arxiv.org/pdf/1809.08895.pdf

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class.

        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length

        """
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)thrshold


        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, hidden_units, dropout_rate, activation='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.activation = activation
        self.w_1 = nn.Linear(idim, hidden_units * 2 if activation == 'glu' else hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)
        if activation == 'learnedswish':
            self.actlayer = LearnedSwish(slope=1)

    def forward(self, x):
        x = self.w_1(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        elif self.activation == 'glu':
            x = F.glu(x)
        elif self.activation == 'softplus':
            x = F.softplus(x)
        elif self.activation == 'swish':
            x = x * torch.sigmoid(x)
        elif self.activation == 'learnedswish':
            x = self.actlayer(x)
        else:
            raise NotImplementedError
        return self.w_2(self.dropout(x))


class LayerNorm(nn.LayerNorm):
    """Layer normalization module

    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization

        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class MultiLayeredConv1d(nn.Module):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-leyered conv1d designed to replace positionwise feed-forward network
    in Transforner block, which is introduced in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    Args:
        in_chans (int): Number of input channels.
        hidden_chans (int): Number of hidden channels.
        kernel_size (int): Kernel size of conv1d.
        dropout_rate (float): Dropout rate.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = nn.Conv1d(in_chans, hidden_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = nn.Conv1d(hidden_chans, in_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, *, in_chans).

        Returns:
            Tensor: Batch of output tensors (B, *, hidden_chans)

        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)

import torch.nn as nn


class VGGish(nn.Module):
    """
    PyTorch implementation of the VGGish model.
    Adapted from: https://github.com/harritaylor/torch-vggish
    The following modifications were made: (i) correction for the missing ReLU layers, (ii) correction for the
    improperly formatted data when transitioning from NHWC --> NCHW in the fully-connected layers, and (iii)
    correction for flattening in the fully-connected layers.
    """

    def __init__(self, idim, odim):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, (1, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2), stride=(1, 2)),

            nn.Conv2d(32, 64, (1, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2), stride=(1, 2)),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 2), stride=(1, 2)),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=3)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * self._calLinDim(idim=idim), odim),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, odim),
            # nn.ReLU(inplace=True),
        )
    def _calLinDim(self,idim):
        for i in range(4):
            idim = (idim - 2) // 2 + 1
        idim = (idim - 3) // 3 + 1

        return idim

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)  # (b, c, t, f)
        # print(f"x.size = {x.size()}")
        x = self.features(x).permute(0, 2, 3, 1).contiguous()
        # print(f"feature.size = {x.size()}")
        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc(x)
        # print(f"fc.size = {x.size()}")
        return x, None

class dynamicConv2d(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate, idfcn):
        super(dynamicConv2d, self).__init__()
        self.dynamicconv = ConvBasis2d(idfcn=idfcn, in_channels=1, out_channels=odim, kernel_size=3, stride=2)


        self.conv = nn.Sequential(
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(odim * (((((idim - 3) // 2 + 1) - 3) // 2) + 1), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask, idw):
        """Subsample x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        # print(f'inputsize = {x.size()}')
        x = x.unsqueeze(1)  # (b, c, t, f)
        # print(f'unseqsize = {x.size()}')
        x = self.dynamicconv(x,idw)
        # print(f'conv = {x.size()}')
        x = self.conv(x)
        # print(f'conv2 = {x.size()}')
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

# dynamic conditional conv layer (it is fc layer when the kernel size is 1x1 and the input is cx1x1)
class ConvBasis2d(nn.Module):
    def __init__(self, idfcn, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, transposed=False, output_padding=_pair(0), groups=1, bias=True):
        super(ConvBasis2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.idfcn = idfcn  # the dimension of coditional input
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.weight_basis = Parameter(torch.Tensor(idfcn*out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(idfcn*out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight_basis.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, input, idw):
        # idw: conditional input
        output = F.conv2d(input, self.weight_basis, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # print(f"weight_basis = {self.weight_basis}")
        # print(self.weight_basis.size())
        # print(idw.size())
        #-(dilation*(kernel-1)+1-1)
        if idw != 'None':
            output = output.view(output.size(0), self.idfcn, self.out_channels, output.size(2), output.size(3)) * \
                     idw.view(-1, self.idfcn, 1, 1, 1).expand(output.size(0), self.idfcn, self.out_channels, output.size(2), output.size(3))
        else:
            output = output.view(output.size(0), self.idfcn, self.out_channels, output.size(2), output.size(3))
        output = output.sum(1).view(output.size(0), output.size(2), output.size(3), output.size(4))

        return output

class Conv2dDilated(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate):
        super(Conv2dDilated, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim//2, 3, dilation=(4,1)),
            nn.ReLU(),
            nn.Conv2d(odim//2, odim, 3, 2, dilation=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2, dilation=(1, 1)),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(odim * (((((idim - 3) // 1 + 1 - 3) // 2 + 1) - 3) // 2 + 1), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):
        """Subsample x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        # print(f'inputsize = {x.size()}')
        x = x.unsqueeze(1)  # (b, c, t, f)
        # print(f'unseqsize = {x.size()}')
        x = self.conv(x)
        # print(f'conv = {x.size()}')
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-8:1][:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate):
        super(Conv2dSubsampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):
        """Subsample x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        # print(f'inputsize = {x.size()}')
        x = x.unsqueeze(1)  # (b, c, t, f)
        # print(f'unseqsize = {x.size()}')
        x = self.conv(x)
        # print(f'conv = {x.size()}')
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsamplingV2(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate=0.0):
        super(Conv2dSubsamplingV2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, odim*2, 3, 2),
            nn.GLU(1),
            nn.Conv2d(odim, odim*2, 3, 2),
            nn.GLU(1)
        )
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):
        """Subsample x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        # print(f'inputsize = {x.size()}')
        x = x.unsqueeze(1)  # (b, c, t, f)
        # print(f'unseqsize = {x.size()}')
        x = self.conv(x)
        # print(f'conv = {x.size()}')
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class LinearWithPosEmbedding(nn.Module):
    def __init__(self, input_size, d_model, dropout_rate=0.0):
        super(LinearWithPosEmbedding, self).__init__()
        self.linear = nn.Linear(input_size, d_model)
        # self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.pos_embedding = PositionalEncoding(d_model, dropout_rate)

    def forward(self, inputs, mask):

        inputs = self.linear(inputs)
        # inputs = self.norm(inputs)
        inputs = self.activation(self.dropout(inputs))
        
        encoded_inputs = self.pos_embedding(inputs)
        return encoded_inputs, mask
        
