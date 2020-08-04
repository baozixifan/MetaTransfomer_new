import os
import torch
import random
import kaldiio as kio
import numpy as np
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader

PAD = 0
EOS = 1
BOS = 1
UNK = 2
MASK = 2
unk = '<unk>'
compute_fbank = ta.compliance.kaldi.fbank

def featureTransform(feature):
    '''
    :param feature:输入特征变量 (...,n_mels,time)
    :return feature:输出特征变量 (time,num_mel_bins)
    '''

    offset = 1e-6 #偏置使log0不是NAN
    if feature.size()[0] != 1:
        feature = feature[0]
    feature = torch.squeeze(feature).permute(1, 0)
    feature = torch.log(feature + offset)

    return feature



def compute_mfcc(wavform, num_mel_bins, sample_frequency):

    feature = ta.transforms.MFCC(sample_rate=sample_frequency, n_mfcc=num_mel_bins, log_mels=True, melkwargs={'n_mels':40,'win_length':322,'hop_length':161})(wavform) #
    if feature.size()[0] != 1:
        feature = feature[0]
    feature = feature.squeeze().transpose(0,1)

    return feature



def load_vocab(vocab_file):
    # unit2idx = {'<S/E>': 0, '<PAD>': 1, '<UNK>': 2}
    unit2idx = {}
    with open(os.path.join(vocab_file), 'r', encoding='utf-8') as v:
        num = 0
        for line in v:
            unit, idx = line.strip().split()
            unit2idx[unit] = int(idx)
            num += 1
    return unit2idx


def normalization(feature):
    # feature = torch.Tensor(feature)
    mean = torch.mean(feature)
    std = torch.std(feature)
    return (feature - mean) / std


def apply_cmvn(mat, stats):
    mean = stats[0, :-1] / stats[0, -1]
    variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
    return np.divide(np.subtract(mat, mean), np.sqrt(variance))


# def time_mask(spec, T=15, num_masks=1, replace_with_zero=False):
#     cloned = spec.clone()
#     len_spectro = cloned.shape[2]

#     for i in range(0, num_masks):
#         t = random.randrange(0, T)
#         t_zero = random.randrange(0, len_spectro - t)

#         # avoids randrange error if values are equal and range is empty
#         if (t_zero == t_zero + t): return cloned

#         mask_end = random.randrange(t_zero, t_zero + t)
#         if (replace_with_zero):
#             cloned[0][:, t_zero:mask_end] = 0
#         else:
#             cloned[0][:, t_zero:mask_end] = cloned.mean()
#     return cloned


# def freq_mask(spec, F=15, num_masks=1, replace_with_zero=False):
#     cloned = spec.clone()
#     num_mel_channels = cloned.shape[1]

#     for i in range(0, num_masks):
#         f = random.randrange(0, F)
#         f_zero = random.randrange(0, num_mel_channels - f)

#         # avoids randrange error if values are equal and range is empty
#         if (f_zero == f_zero + f): return cloned

#         mask_end = random.randrange(f_zero, f_zero + f)
#         if (replace_with_zero):
#             cloned[0][f_zero:mask_end] = 0
#         else:
#             cloned[0][f_zero:mask_end] = cloned.mean()

#     return cloned

#频谱增强
def spec_augment(mel_spectrogram, frequency_mask_num=1, time_mask_num=2,
                 frequency_masking_para=5, time_masking_para=15):
    tau = mel_spectrogram.shape[0]
    v = mel_spectrogram.shape[1]

    warped_mel_spectrogram = mel_spectrogram

    # Step 2 : Frequency masking
    if frequency_mask_num > 0:
        for i in range(frequency_mask_num):
            f = np.random.uniform(low=0.0, high=frequency_masking_para)
            f = int(f)
            f0 = random.randint(0, v-f)
            warped_mel_spectrogram[:, f0:f0+f] = 0

    # Step 3 : Time masking
    if time_mask_num > 0:
        for i in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_masking_para)
            t = int(t)
            t0 = random.randint(0, tau-t)
            warped_mel_spectrogram[t0:t0+t, :] = 0

    return warped_mel_spectrogram


def concat_and_subsample(features, left_frames=3, right_frames=0, skip_frames=2):

    time_steps, feature_dim = features.shape
    concated_features = np.zeros(
        shape=[time_steps, (1+left_frames+right_frames) * feature_dim], dtype=np.float32)

    concated_features[:, left_frames * feature_dim: (left_frames+1)*feature_dim] = features

    for i in range(left_frames):
        concated_features[i+1: time_steps, (left_frames-i-1)*feature_dim: (
            left_frames-i) * feature_dim] = features[0:time_steps-i-1, :]

    for i in range(right_frames):
        concated_features[0:time_steps-i-1, (right_frames+i+1)*feature_dim: (
            right_frames+i+2)*feature_dim] = features[i+1: time_steps, :]

    return concated_features[::skip_frames+1, :]


class AudioDataset(Dataset):
    def __init__(self, params, name='train'):

        self.name = name
        self.params = params
        self.left_frames = params['left_frames']
        self.right_frames = params['right_frames']
        self.skip_frames = params['skip_frames']

        self.unit2idx = load_vocab(params['vocab'])
        # print(self.unit2idx)

        if params['from_kaldi']:
            self.from_kaldi = True
            print('Load Kaldi Features!')
        else:
            self.from_kaldi = False
            print('Extract Features ONLINE!')

        self.targets_dict = {}
        with open(os.path.join(params[name], params['text']), 'r', encoding='utf-8') as t:
            for line in t:
                parts = line.strip().split()
                # print(f"parts = {parts}")
                utt_id = parts[0]
                label = []
                # print(parts[1])
                for c in parts[1]:
                    label.append(self.unit2idx[c] if c in self.unit2idx else self.unit2idx[unk])
                # print(label)
                self.targets_dict[utt_id] = label
        # print(self.targets_dict)

        self.file_list = []
        with open(os.path.join(params[name], 'feats.scp' if self.from_kaldi else 'wav.scp'), 'r', encoding='utf-8') as fid:
            for line in fid:
                idx, path = line.strip().split()
                self.file_list.append([idx, path])
        # print(self.file_list)

        assert len(self.file_list) == len(
            self.targets_dict), 'please keep feats.scp and %s have the same lines.' % params['text']

        self.lengths = len(self.file_list)

        if params['apply_cmvn']:

            assert os.path.isfile(os.path.join(params[name], 'utt2spk'))
            assert os.path.isfile(os.path.join(params[name], 'cmvn.scp'))

            self.utt2spk = {}
            with open(os.path.join(params[name], 'utt2spk'), 'r') as f:
                for line in f:
                    utt_id, spk_id = line.strip().split()
                    self.utt2spk[utt_id] = spk_id
                print('Load Speaker INFO')

            self.cmvns = {}
            with open(os.path.join(params[name], 'cmvn.scp'), 'r') as f:
                for line in f:
                    spk_id, path = line.strip().split()
                    self.cmvns[spk_id] = path
                print('Load CMVN Stats')

        if self.params['spec_augment'] and self.name == 'train':
            # self.TimeMasking = ta.transforms.TimeMasking(time_mask_param=15, iid_masks=False)
            # self.FrequencyMasking = ta.transforms.FrequencyMasking(freq_mask_param=5, iid_masks=False)

            print('Apply spec_augment!')

    def __getitem__(self, index):
        utt_id, path = self.file_list[index]

        if self.from_kaldi:
            feature = kio.load_mat(path)
        else:
            #TODO 去掉
            # print(f"path = {path}")
            try:
                wavform, sample_frequency = ta.load_wav(path)

                feature = compute_fbank(wavform, num_mel_bins=self.params['num_mel_bins'],
                                        sample_frequency=sample_frequency)
                # feature = ta.compliance.kaldi.spectrogram(wavform,
                #                                           dither=0.1,
                #                                           sample_frequency=sample_frequency)

                # feature2 = ta.compliance.kaldi.mfcc(wavform, num_ceps=40, num_mel_bins=self.params['num_mel_bins'],
                #                                    sample_frequency=sample_frequency)
                # feature = torch.cat((feature,feature2), dim=1)

                # feature = compute_mfcc(wavform, num_mel_bins=self.params['num_mel_bins'],
                #                        sample_frequency=sample_frequency)

                # mel_specgram = ta.transforms.MelSpectrogram(win_length=322,
                #                                             hop_length=161,
                #                                             n_mels=self.params['num_mel_bins'],
                #                                             sample_rate=sample_frequency)(wavform)
                # if self.params['spec_argument'] and self.name == 'train':
                #     mel_specgram = self.TimeMasking(specgram=mel_specgram)
                #     mel_specgram = self.FrequencyMasking(specgram=mel_specgram)
                #
                # feature = featureTransform(mel_specgram)

                # print(f'feature = {feature.size()}')
                # print(f'feature2 = {feature2.size()}')
                # print(f'feature3 = {feature3.size()}')
            except:
                print(f"path = {path}")

        if self.params['apply_cmvn']:
            spk_id = self.utt2spk[utt_id]
            stats = kio.load_mat(self.cmvns[spk_id])
            feature = apply_cmvn(feature, stats)

        if self.params['normalization']:
            feature = normalization(feature)
            # print(f"featuretype1 = {feature.type()}")

            # print(f"featuretype2 = {feature.type()}")
        #TODO 测试语音增强
        if self.params['spec_augment'] and self.name == "train":
            try:
                feature = spec_augment(feature)
                # print("1"*50)
            except:
                # print("*"*50)
                pass

        if self.left_frames > 0 or self.right_frames > 0:
            feature = concat_and_subsample(feature, left_frames=self.left_frames,
                                           right_frames=self.right_frames, skip_frames=self.skip_frames)

        feature_length = feature.shape[0]
        targets = self.targets_dict[utt_id]
        targets_length = len(targets)
        #TODO 去掉
        # print(f"utt_id = {utt_id}")
        # print(f"feature = {feature}")


        return utt_id, feature, feature_length, targets, targets_length



    def __len__(self):
        return self.lengths

    def read_features(self, path):
        raise NotImplementedError

    def encode(self, seq):

        encoded_seq = []
        if self.encoding:
            for c in seq:
                if c in self.unit2idx:
                    encoded_seq.append(self.unit2idx[c])
                else:
                    encoded_seq.append(self.unit2idx['<UNK>'])
        else:
            encoded_seq = [int(i) for i in seq]

        return encoded_seq

    @property
    def idx2char(self):
        return {i: c for (c, i) in self.unit2idx.items()}

    @property
    def vocab_size(self):
        return len(self.unit2idx)

    @property
    def batch_size(self):
        return self.params['batch_size']


def collate_fn_with_eos_bos(batch):

    utt_ids = [data[0] for data in batch]
    features_length = [data[2] for data in batch]
    targets_length = [data[4] for data in batch]
    max_feature_length = max(features_length)
    max_target_length = max(targets_length)

    padded_features = []
    padded_targets = []

    for _, feat, feat_len, target, target_len in batch:
        padded_features.append(np.pad(feat, ((
            0, max_feature_length-feat_len), (0, 0)), mode='constant', constant_values=0.0))
        padded_targets.append(
            [BOS] + target + [EOS] + [PAD] * (max_target_length - target_len))

    features = torch.FloatTensor(padded_features)
    features_length = torch.IntTensor(features_length)
    targets = torch.LongTensor(padded_targets)
    targets_length = torch.IntTensor(targets_length)

    inputs = {
        'inputs': features,
        'inputs_length': features_length,
        'targets': targets,
        'targets_length': targets_length
    }
    return utt_ids, inputs


def collate_fn(batch):

    utt_ids = [data[0] for data in batch]
    features_length = [data[2] for data in batch]
    targets_length = [data[4] for data in batch]
    max_feature_length = max(features_length)
    max_target_length = max(targets_length)

    padded_features = []
    padded_targets = []

    for _, feat, feat_len, target, target_len in batch:
        padded_features.append(np.pad(feat, ((
            0, max_feature_length-feat_len), (0, 0)), mode='constant', constant_values=0.0))
        padded_targets.append(target + [PAD] * (max_target_length - target_len))

    features = torch.FloatTensor(padded_features)
    features_length = torch.IntTensor(features_length)
    targets = torch.LongTensor(padded_targets)
    targets_length = torch.IntTensor(targets_length)

    feature = {
        'inputs': features,
        'inputs_length': features_length
    }

    label = {
        'targets': targets,
        'targets_length': targets_length
    }
    return utt_ids, feature, label


class FeatureLoader(object):
    def __init__(self, dataset, shuffle=False, ngpu=1, mode='ddp', include_eos_sos=True):
        if ngpu > 1:
#             if mode == 'hvd':
#                 import horovod.torch as hvd
#                 self.sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(),
#                                                                                rank=hvd.rank())
            if mode == 'ddp':
                self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                self.sampler = None
        else:
            self.sampler = None

        self.loader = torch.utils.data.DataLoader(dataset, batch_size=dataset.batch_size * ngpu,
                                                  shuffle=shuffle if self.sampler is None else False,
                                                  num_workers=2 * ngpu, pin_memory=False, sampler=self.sampler,
                                                  collate_fn=collate_fn_with_eos_bos if include_eos_sos else collate_fn)
        # self.loader = torch.utils.data.DataLoader(dataset, batch_size=dataset.batch_size * ngpu,
        #                                           shuffle= False,
        #                                           num_workers=2 * ngpu, pin_memory=False, sampler=self.sampler,
        #                                           collate_fn=collate_fn_with_eos_bos if include_eos_sos else collate_fn)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
