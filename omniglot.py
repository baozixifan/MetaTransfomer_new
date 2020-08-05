import  torch.utils.data as data
import  os
import  os.path
import  errno

import torch
import random
import numpy as np
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler

PAD = 0
EOS = 1
BOS = 1
UNK = 2
MASK = 2
unk = '<unk>'
compute_fbank = ta.compliance.kaldi.fbank

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


class AudioDatasetTest(Dataset):
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

            except:
                print(f"path = {path}")

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


def collate_fn_with_eos_bos_test(batch):

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
    return inputs


class FeatureLoaderTest(object):
    def __init__(self, dataset, shuffle=False, ngpu=1, mode='ddp', testBatch=10):

        self.sampler = None
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=testBatch,
                                                  shuffle=shuffle if self.sampler is None else False,
                                                  num_workers=2 * ngpu, pin_memory=False, sampler=self.sampler,
                                                  collate_fn=collate_fn_with_eos_bos_test)





class AudioDataset(Dataset):
    def __init__(self, params, name='train'):

        self.name = name
        self.params = params
        self.unit2idx = load_vocab(params['vocab'])
        # print(self.unit2idx)

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
        self.datasetSplit = {}
        num = 0
        idxTmp = ''
        with open(os.path.join(params[name], 'wav.scp'), 'r', encoding='utf-8') as fid:
            for line in fid:
                idx, path = line.strip().split()
                if idxTmp != idx[:-5]:
                    idxTmp = idx[:-5]
                    start = num
                    self.datasetSplit[idxTmp] = start
                self.file_list.append([idx, path])
                num += 1
        # print(self.datasetSplit)
        print(f"len_datasetSplit = {len(self.datasetSplit)}")

        assert len(self.file_list) == len(
            self.targets_dict), 'please keep feats.scp and %s have the same lines.' % params['text']

        self.lengths = len(self.file_list)


    def __getitem__(self, index):

        utt_id, path = self.file_list[index]

        #TODO 去掉
        # print(f"path = {path}")
        try:
            wavform, sample_frequency = ta.load_wav(path)

            feature = compute_fbank(wavform, num_mel_bins=self.params['num_mel_bins'],
                                    sample_frequency=sample_frequency)

            # feature = featureTransform(mel_specgram)
        except:
            print(f"path = {path}")

        if self.params['normalization']:
            feature = normalization(feature)
            # print(f"featuretype1 = {feature.type()}")

        #TODO 测试语音增强
        if self.params['spec_augment'] and self.name == "train":
            try:
                feature = spec_augment(feature)
                # print("1"*50)
            except:
                # print("*"*50)
                pass


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

def collate_fn_with_eos_bos_indepTask(batch):

    k_shot = 7
    k_query = 7
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

    features = padded_features
    targets = padded_targets
    features_length = np.array(features_length)
    targets_length = np.array(targets_length)

    #划分support和query集
    utt_ids_spt = utt_ids[:k_shot]
    features_spt = np.array(features[:k_shot]).reshape(k_shot, -1, 40)
    features_spt_length = np.array(features_length[:k_shot])
    targets_spt = np.array(targets[:k_shot]).reshape(k_shot, -1)
    targets_spt_length = np.array(targets_length[:k_shot])

    utt_ids_qry = utt_ids[k_shot:]
    features_qry = np.array(features[k_shot:]).reshape(k_query, -1, 40)
    features_qry_length = np.array(features_length[k_shot:])
    targets_qry = np.array(targets[k_shot:]).reshape(k_query, -1)
    targets_qry_length = np.array(targets_length[k_shot:])


    return utt_ids_spt, features_spt, features_spt_length, targets_spt, targets_spt_length, utt_ids_qry, features_qry, features_qry_length, targets_qry, targets_qry_length


def collate_fn_with_eos_bos(batch):


    utt_ids = [data[0] for data in batch]
    features_length = [data[2] for data in batch]
    targets_length = [data[4] for data in batch]

    padded_features = [data[1] for data in batch]
    padded_targets = [data[3] for data in batch]

    features = padded_features
    features_length = features_length
    targets = padded_targets
    targets_length = targets_length

    #划分support和query集
    utt_ids_spt = utt_ids[:3]
    features_spt = features[:3]
    features_spt_length = features_length[:3]
    targets_spt = targets[:3]
    targets_spt_length = targets_length[:3]

    utt_ids_qry = utt_ids[3:]
    features_qry = features[3:]
    features_qry_length = features_length[3:]
    targets_qry = targets[3:]
    targets_qry_length = targets_length[3:]

    # query = {
    #     "utt_ids": utt_ids_qry,
    #     'features': features_qry,
    #     'features_length': features_qry_length,
    #     'targets': targets_qry,
    #     'targets_length': targets_qry_length
    # }
    support = []
    support.append(utt_ids_spt)
    support.append(features_spt)
    support.append(features_spt_length)
    support.append(targets_spt)
    support.append(targets_spt_length)

    query = []
    query.append(utt_ids_qry)
    query.append(features_qry)
    query.append(features_qry_length)
    query.append(targets_qry)
    query.append(targets_qry_length)

    return support, query



class MySubsetSampler(SubsetRandomSampler):


    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))




class FeatureLoader(object):
    def __init__(self, dataset, shuffle=False, ngpu=1, mode='ddp'):

        keys = list(dataset.datasetSplit.keys())
        lenKeys = len(keys)

        indicesAll = list(range(len(dataset)))


        self.samplerTask1 = MySubsetSampler(indicesAll[dataset.datasetSplit[keys[0]]:dataset.datasetSplit[keys[113]]])
        self.samplerTask2 = MySubsetSampler(indicesAll[dataset.datasetSplit[keys[113]]:dataset.datasetSplit[keys[226]]])
        self.samplerTask3 = MySubsetSampler(indicesAll[dataset.datasetSplit[keys[226]]:dataset.datasetSplit[keys[339]]])


        self.loaderTask1 = torch.utils.data.DataLoader(dataset, batch_size=dataset.batch_size,
                                                  shuffle=False,drop_last=True,
                                                  num_workers=2 * ngpu, pin_memory=False, sampler=self.samplerTask1,
                                                  collate_fn=collate_fn_with_eos_bos_indepTask)
        self.loaderTask2 = torch.utils.data.DataLoader(dataset, batch_size=dataset.batch_size,
                                                  shuffle=False, drop_last=True,
                                                  num_workers=2 * ngpu, pin_memory=False, sampler=self.samplerTask2,
                                                  collate_fn=collate_fn_with_eos_bos_indepTask)
        self.loaderTask3 = torch.utils.data.DataLoader(dataset, batch_size=dataset.batch_size,
                                                  shuffle=False, drop_last=True,
                                                  num_workers=2 * ngpu, pin_memory=False, sampler=self.samplerTask3,
                                                  collate_fn=collate_fn_with_eos_bos_indepTask)


def lengthTrans(listTest):
    listTmp = []
    listTest = np.array(listTest)
    for i in range(listTest.shape[0]):
        for j in range(listTest.shape[1]):
            listTmp.append(listTest[i][j])
    return listTmp



def appendSet(listSet:list, n_way, k_shot_query):

    listLen = len(listSet)
    utt_ids = []
    features = []
    features_length = []
    targets = []
    targets_length = []
    for i in listSet:
        utt_ids.append(i[0])
        features.append(i[1])
        features_length.append(i[2])
        targets.append(i[3])
        targets_length.append(i[4])

    features_length = lengthTrans(features_length)
    targets_length = lengthTrans(targets_length)

    max_feature_length = max(features_length)
    max_target_length = max(targets_length)

    padded_features = []
    padded_targets = []

    for i in range(listLen):
        for j in range(k_shot_query):
            padded_features.append(np.pad(features[i][j], ((
                0, max_feature_length-features_length[i * k_shot_query + j]), (0, 0)), mode='constant', constant_values=0.0))
            # print(padded_features)
            # print(padded_features[j].shape)
            padded_targets.append(
                np.array([BOS] + targets[i][j] + [EOS] + [PAD] * (max_target_length - targets_length[i * k_shot_query + j]))
            )

    # for i in range(listLen*k_shot_query):
    #     print(padded_features[i].shape)
    # print(np.array(padded_features).shape)
    # print(np.array(padded_features))
    # print(len(padded_features))
    # print(padded_features[40].shape)
    #reshape self.n_way * self.k_shot
    #TODO:feature维度
    features = np.array(padded_features).reshape(n_way * k_shot_query, -1, 40)
    targets = np.array(padded_targets).reshape(n_way * k_shot_query, -1)
    features_length = np.array(features_length)
    targets_length = np.array(targets_length)

    # print(f"features = {features}")
    # print(f"features_length = {features_length}")
    # print(f"targets = {targets}")
    # print(f"targets_length = {targets_length}")

    return utt_ids, features, features_length, targets, targets_length

def save_model(model, expdir, epoch=None, save_name=None):
    if save_name is None:
        save_name = 'model.temp.%d.pt' % epoch

    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict()
    }

    torch.save(checkpoint, expdir + '/' + save_name)

def load_model(model, checkpoint):

    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict['model'])

if __name__ == "__main__":
    import yaml
    from meta import Meta
    import argparse
    from itertools import cycle
    from otrans.utils import map_to_cuda, init_logger, AverageMeter, Summary
    from copy import deepcopy

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=400)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=3)

    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=3)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    argparser.add_argument('-c', '--config', type=str, default=None)
    argparser.add_argument('-n', '--ngpu', type=int, default=1)
    argparser.add_argument('-s', '--seed', type=int, default=1222)
    argparser.add_argument('-p', '--parallel_mode', type=str, default='dp')
    argparser.add_argument('-r', '--local_rank', type=int, default=0)

    args = argparser.parse_args()

    with open(r"egs/aishellorg/conf/transformer.yaml", 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    expdir = os.path.join('egs', params['data']['name'], 'exp', params['train']['save_name'])
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    logger = init_logger(log_file=os.path.join(expdir, 'train.log'))


    train_dataset = AudioDataset(params['data'], 'train')

    train_loader = FeatureLoader(train_dataset, shuffle=False, ngpu=1,
                                 mode='dp')

    maml = Meta(args, params)
    # task1 = iter(train_loader.loaderTask1)
    # task2 = iter(train_loader.loaderTask2)
    # task3 = iter(train_loader.loaderTask3)
    #
    # for i in range(50):
    #     print(f"{i}************************************************************")
    #     (supportTask1, queryTask1) = task1.next()
    #     print(f"supportTask1 = {supportTask1}")
    #     # print(f"queryTask1 = {queryTask1}")
    #
    #     (supportTask2, queryTask2) = task2.next()
    #     print(f"supportTask2 = {supportTask2}")
    #     # print(f"queryTask2 = {queryTask2}")
    #
    #     (supportTask3, queryTask3) = task3.next()
    #     print(f"supportTask2 = {supportTask3}")
    #     # print(f"queryTask2 = {queryTask3}")

    utt_ids_spts = []
    features_spts = []
    features_length_spts = []
    targets_spts = []
    targets_length_spts = []

    utt_ids_qrys = []
    features_qrys = []
    features_length_qrys = []
    targets_qrys = []
    targets_length_qrys = []

    testLossNote = Summary()

    for epoch in range(args.epoch):

        for step, data in enumerate(zip(train_loader.loaderTask1, train_loader.loaderTask2, train_loader.loaderTask3)):
            # print(f"step = {step}")
            # print(f"utt_ids = {utt_ids}")

            # (supportTask1, queryTask1) = data[0]
            # (supportTask2, queryTask2) = data[1]
            # (supportTask3, queryTask3) = data[2]
            #
            # print("*" * 50)
            # print(f"supportTask1 = {supportTask1}")
            # print("*" * 50)
            # print(f"supportTask2 = {supportTask2}")
            # print("*" * 50)
            # print(f"supportTask3 = {supportTask3}")
            for i in range(args.task_num):
                utt_ids_spt, features_spt, features_length_spt, targets_spt, targets_length_spt,\
                utt_ids_qry, features_qry, features_length_qry, targets_qry, targets_length_qry = data[i]

                utt_ids_spts.append(utt_ids_spt)
                features_spts.append(features_spt)
                features_length_spts.append(features_length_spt)
                targets_spts.append(targets_spt)
                targets_length_spts.append(targets_length_spt)

                utt_ids_qrys.append(utt_ids_qry)
                features_qrys.append(features_qry)
                features_length_qrys.append(features_length_qry)
                targets_qrys.append(targets_qry)
                targets_length_qrys.append(targets_length_qry)




            # print(f"features_spts = {features_spts}")
            # print(f"features_length_spts = {features_length_spts}")
            # print(f"targets_spts = {targets_spts}")
            # print(f"targets_length_spts = {targets_length_spts}")

            support = {
                "features_spts": features_spts,
                "features_length_spts": features_length_spts,
                "targets_spts": targets_spts,
                "targets_length_spts": targets_length_spts
            }
            query = {
                "features_qrys": features_qrys,
                "features_length_qrys": features_length_qrys,
                "targets_qrys": targets_qrys,
                "targets_length_qrys": targets_length_qrys
            }

            loss = maml(support, query)
            print('step:', step, '\ttraining loss:', loss.item())

            utt_ids_spts = []
            features_spts = []
            features_length_spts = []
            targets_spts = []
            targets_length_spts = []

            utt_ids_qrys = []
            features_qrys = []
            features_length_qrys = []
            targets_qrys = []
            targets_length_qrys = []

        if (epoch) % 1 == 0:
            logger.info(f"'epoch:', {epoch}, 'training loss:', {loss}")

        if (epoch) % 1 == 0:
            # torch.cuda.empty_cache()
            fast_model = deepcopy(maml.net)
            fast_model.eval()

            lossTest = 0

            test_dataset = AudioDatasetTest(params['data'], 'test')

            test_loader = FeatureLoaderTest(test_dataset, shuffle=False, ngpu=1,
                                         mode='dp')
            for stepInner, batch in enumerate(test_loader.loader):
                batch = map_to_cuda(batch)
                features, features_length, targets, targets_length = \
                    batch['inputs'], batch['inputs_length'], batch['targets'], batch['targets_length']

                testLoss = maml.finetunning(features, features_length, targets, targets_length, fast_model)
                # lossTest.append(testLoss)
                if (stepInner+1) % 200 == 0:
                    logger.info(f"test : epochTest = 1 ; stepInner = {stepInner} ; testLoss = {testLoss}")
                lossTest += testLoss.item()

            lossTestAvg = lossTest / stepInner
            logger.info(f"testLoss average = {lossTestAvg}")
            testLossNote.update(epoch, lossTestAvg)

            if lossTestAvg <= testLossNote.best()[1]:
                logger.info(f'Update the best checkpoint! epoch = {epoch}')

            save_name = 'model.temp.%d.pt' % epoch
            logger.info(f'save the model : {save_name}')
            save_model(maml.net, expdir, epoch=epoch, save_name=None)

            del fast_model

            # torch.cuda.empty_cache()







