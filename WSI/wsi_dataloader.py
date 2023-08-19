import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils import *


def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats


class C16DatasetV3(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.split = split
        if split == 'train':
            self.data_csv = pd.read_csv(join(args.dataroot, 'train_offical.csv'))
        elif split == 'test':
            self.data_csv = pd.read_csv(join(args.dataroot, 'test_offical.csv'))
        elif split =='val':
            self.data_csv = pd.read_csv(join(args.dataroot, 'val_offical.csv'))
            # drop_idx = []
            # for i in range(len(self.data_csv)):
            #     if self.data_csv.iloc[i, 0] in ['test_114', 'test_124']:
            #         drop_idx.append(i)
            # self.data_csv.drop(drop_idx, axis=0, inplace=True)
            # self.data_csv = self.data_csv.reset_index(drop=True)

        if isdir(join(args.dataroot, 'single_b' + str(args.backgrd_thres))):
            func = lambda row: join(args.dataroot, 'single_b' + str(args.backgrd_thres), row + ".csv")
        else:
            func = lambda row: join(args.dataroot, 'feats', row + ".csv")

        self.data_csv['subject_id'] = self.data_csv['subject_id'].apply(func)

    def get_bag_feats(self, csv_file_df):
        feats_csv_path = csv_file_df.iloc[0]

        df = pd.read_csv(feats_csv_path)
        feats = shuffle(df).reset_index(drop=True)
        feats = feats.to_numpy()
        label = np.zeros(self.args.num_classes)
        if self.args.num_classes==1:
            label[0] = csv_file_df.iloc[1]
        else:
            if int(csv_file_df.iloc[1])<=(len(label)-1):
                label[int(csv_file_df.iloc[1])] = 1

        label = torch.tensor(np.array(label))
        feats = torch.tensor(np.array(feats)).float()
        return label, feats
    
    def __getitem__(self, idx):
        label, feats = self.get_bag_feats(self.data_csv.iloc[idx])

        if self.split == 'train' and self.args.dropout_patch > 0.0:
            #print('drop')
            feats = dropout_patches(feats, self.args.dropout_patch)

        return feats, label
        
    def __len__(self):
        return len(self.data_csv)

class C16DatasetV2(Dataset):
    def __init__(self, dataroot, split='train', split_ratio=0.9, num_classes=2, dropout_patch=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_patch = dropout_patch
        self.split = split

        if split in ['train', 'val']:
            with open(join(dataroot, 'mDATA_train.pkl'), 'rb') as f:
                mDATA = pickle.load(f)

            mDATA_train , mDATA_val  = [i.to_dict() for i in train_test_split(pd.Series(mDATA), train_size=split_ratio, random_state=42)]

            if split == 'train':
                mDATA = mDATA_train
            else:
                mDATA = mDATA_val

            SlideNames, FeatList, Label = self.reOrganize_mDATA(mDATA)
        elif split == 'test':
            with open(join(dataroot, 'mDATA_test.pkl'), 'rb') as f:
                mDATA = pickle.load(f)
            test_info = pd.read_csv(join(dataroot, 'reference.csv'))
            SlideNames, FeatList, Label = self.reOrganize_mDATA_test(mDATA, test_info)

        self.SlideNames = SlideNames
        self.FeatList = FeatList
        self.Label = Label

    def reOrganize_mDATA_test(self, mDATA, test_info):
        tumorSlides = []
        for i in range(len(test_info)):
            if test_info.iloc[i, 1] == 'Tumor':
                tumorSlides.append(test_info.iloc[i, 0])

        SlideNames = []
        FeatList = []
        Label = []
        for slide_name in mDATA.keys():
            SlideNames.append(slide_name)

            if slide_name in tumorSlides:
                label = 1
            else:
                label = 0
            Label.append(label)

            patch_data_list = mDATA[slide_name]
            featGroup = []
            for tpatch in patch_data_list:
                tfeat = torch.from_numpy(tpatch['feature'])
                featGroup.append(tfeat.unsqueeze(0))
            featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
            FeatList.append(featGroup)

        return SlideNames, FeatList, Label

    def reOrganize_mDATA(self, mDATA):
        SlideNames = []
        FeatList = []
        Label = []
        for slide_name in mDATA.keys():
            SlideNames.append(slide_name)

            if slide_name.startswith('tumor'):
                label = 1
            elif slide_name.startswith('normal'):
                label = 0
            else:
                raise RuntimeError('Undefined slide type')
            Label.append(label)

            patch_data_list = mDATA[slide_name]
            featGroup = []
            for tpatch in patch_data_list:
                tfeat = torch.from_numpy(tpatch['feature'])
                featGroup.append(tfeat.unsqueeze(0))
            featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
            FeatList.append(featGroup)

        return SlideNames, FeatList, Label

    def __len__(self):
        return len(self.SlideNames)

    def __getitem__(self, idx):
        slide_name = self.SlideNames[idx]
        if self.dropout_patch > 0.0 and self.split == 'train':
            bag_feat = dropout_patches(self.FeatList[idx], self.dropout_patch)

        bag_feat = shuffle(bag_feat)
        bag_label = int(self.Label[idx])

        label = np.zeros(self.num_classes)
        if self.num_classes==1:
            label[0] = bag_label
        else:
            if bag_label <= (len(label) - 1):
                label[bag_label] = 1
        
        return slide_name, bag_feat, label
    

class C16DatasetV1(Dataset):
    def __init__(self, dataroot, split="train", level=1, onehot_label=True, dropout_patch_rate=0.0, seed=0):
        super().__init__()
        self.split = split
        self.onehot_label = onehot_label
        self.dropout_patch_rate = dropout_patch_rate
        self.seed = seed
        self.base_dir = join(dataroot, split, f"{10.0 * level:.1f}", "extracted_features")   
        self.csv = shuffle(pd.read_csv(join(dataroot, split + "_offical.csv")).sort_values(by=['subject_id']), random_state=seed)
        self.bag_labels = self.csv.iloc[:, -1].values.tolist()

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        subject_id = self.csv.iloc[idx, 0]
        bag_label = int(self.csv.iloc[idx, 1])

        if self.split == 'train':
            data_file = join(self.base_dir, "normal" if bag_label == 0 else "tumor", subject_id + ".npy")
        elif self.split == 'test':
            data_file = join(self.base_dir, subject_id + ".npy")

        embed_feats = np.load(data_file)

        if self.onehot_label:
            bag_label_new = np.zeros(2, dtype=np.int32)
            bag_label_new[bag_label] = 1
            bag_label = bag_label_new
        
        embed_feats = self.augment(embed_feats)

        if self.split == 'train':
            if self.dropout_patch_rate and self.dropout_patch_rate > 0.0:
                embed_feats = dropout_patches(embed_feats, self.dropout_patch_rate)

        return embed_feats, bag_label
    
    def augment(self, feats):
        np.random.shuffle(feats)
        return feats


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = C16DatasetV2("datasets/Camelyon16/imagenet-resnet50", "val")
    dataloader = DataLoader(dataset, 1, True, drop_last=False)
    
    for slide_name, bag_feat, bag_label in dataloader:
        print(slide_name, bag_label)
        print(bag_feat.size())

        break