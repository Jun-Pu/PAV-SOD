import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torchaudio
import cv2
import torch

clip_length = 3

def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


# dataset for training
class SalObjDataset(data.Dataset):
    def __init__(self, root, trainsize):
        self.trainsize = trainsize

        # get visual and audio clips (duration of three consecutive key frames)
        with open(root, 'r') as f:
            self.seqs = [x.strip() for x in f.readlines()]

        video_clips, gt_clips = [], []
        audio_clips_ch1, audio_clips_ch2, audio_clips_ch3, audio_clips_ch4 = [], [], [], []  # ambisonics
        for seq_idx in self.seqs:
            # get visual clips of each video
            frm_list = os.listdir(os.path.join('/home/yzhang1/PythonProjects/AV360/frame_key/', seq_idx))
            frm_list = sorted(frm_list)
            gt_list = os.listdir(os.path.join('/home/yzhang1/PythonProjects/AV360/DATA/train/', seq_idx))
            gt_list = sorted(gt_list)
            for idx in range(len(frm_list)):
                frm_list[idx] = '/home/yzhang1/PythonProjects/AV360/frame_key/' + seq_idx + '/' + frm_list[idx]
                gt_list[idx] = '/home/yzhang1/PythonProjects/AV360/DATA/train/' + seq_idx + '/' + gt_list[idx]
            frm_list_split = list(split_list(frm_list, int(len(frm_list) / clip_length)))
            gt_list_split = list(split_list(gt_list, int(len(gt_list) / clip_length)))
            video_clips.append(frm_list_split)
            gt_clips.append(gt_list_split)

            # get audio clips of each video
            audio_pth = '/home/yzhang1/PythonProjects/AV360/ambisonics_trimmed/' + seq_idx + '.wav'
            audio_ori = torchaudio.load(audio_pth)[0]
            audio_ori_ch1 = audio_ori[0]
            audio_ori_ch2 = audio_ori[1]
            audio_ori_ch3 = audio_ori[2]
            audio_ori_ch4 = audio_ori[3]
            audio_split_size = int(len(audio_ori[1])/(int(len(frm_list) / clip_length)))
            audio_ch1_split = torch.split(tensor=audio_ori_ch1, split_size_or_sections=audio_split_size)
            audio_ch1_split = list(audio_ch1_split)
            audio_ch1_split = audio_ch1_split[:int(len(frm_list) / clip_length)]
            audio_clips_ch1.append(audio_ch1_split)
            audio_ch2_split = torch.split(tensor=audio_ori_ch2, split_size_or_sections=audio_split_size)
            audio_ch2_split = list(audio_ch2_split)
            audio_ch2_split = audio_ch2_split[:int(len(frm_list) / clip_length)]
            audio_clips_ch2.append(audio_ch2_split)
            audio_ch3_split = torch.split(tensor=audio_ori_ch3, split_size_or_sections=audio_split_size)
            audio_ch3_split = list(audio_ch3_split)
            audio_ch3_split = audio_ch3_split[:int(len(frm_list) / clip_length)]
            audio_clips_ch3.append(audio_ch3_split)
            audio_ch4_split = torch.split(tensor=audio_ori_ch4, split_size_or_sections=audio_split_size)
            audio_ch4_split = list(audio_ch4_split)
            audio_ch4_split = audio_ch4_split[:int(len(frm_list) / clip_length)]
            audio_clips_ch4.append(audio_ch4_split)

        self.video_clips_flatten = [item for sublist in video_clips for item in sublist]
        self.gt_clips_flatten = [item for sublist in gt_clips for item in sublist]

        self.audio_ch1_clips_flatten = [item for sublist in audio_clips_ch1 for item in sublist]
        self.audio_ch2_clips_flatten = [item for sublist in audio_clips_ch2 for item in sublist]
        self.audio_ch3_clips_flatten = [item for sublist in audio_clips_ch3 for item in sublist]
        self.audio_ch4_clips_flatten = [item for sublist in audio_clips_ch4 for item in sublist]

        # tools for data transform
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize * 2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize * 2)),
            transforms.ToTensor()])

        # dataset size
        self.size = len(self.video_clips_flatten)

    def __getitem__(self, index):
        seq_name = self.video_clips_flatten[index][0].split('/')[-2]
        imgs, gts, audios = [], [], []
        for idx in range(clip_length):  # default as 3 consecutive frames of each clip
            curr_img = self.rgb_loader(self.video_clips_flatten[index][idx])
            imgs.append(self.img_transform(curr_img))

            curr_gt = self.binary_loader(self.gt_clips_flatten[index][idx])
            gts.append(self.gt_transform(curr_gt))

        # collect audio clip  (default as 15 frames equivalently; five times of clip length)
        if index == 0: a_index = [0, 1, 2, 3, 4]
        elif index == 1: a_index = [0, 1, 2, 3, 4]
        elif index == self.size - 1:
            a_index = [self.size - 5, self.size - 4, self.size - 3, self.size - 2, self.size - 1]
        elif index == self.size - 2:
            a_index = [self.size - 5, self.size - 4, self.size - 3, self.size - 2, self.size - 1]
        else:
            if self.video_clips_flatten[index - 2][0].split('/')[-2] != seq_name:
                if self.video_clips_flatten[index - 1][0].split('/')[-2] != seq_name:
                    a_index = [index, index + 1, index + 2, index + 3, index + 4]
                else:
                    a_index = [index - 1, index, index + 1, index + 2, index + 3]
            elif self.video_clips_flatten[index + 2][0].split('/')[-2] != seq_name:
                if self.video_clips_flatten[index + 1][0].split('/')[-2] != seq_name:
                    a_index = [index - 4, index - 3, index - 2, index - 1, index]
                else:
                    a_index = [index - 3, index - 2, index - 1, index, index + 1]
            else:
                    a_index = [index - 2, index - 1, index, index + 1, index + 2]

        a_ch1, a_ch2, a_ch3, a_ch4 = [], [], [], []
        for aa in range(5):
            a_ch1.append(self.audio_ch1_clips_flatten[a_index[aa]])
            a_ch2.append(self.audio_ch2_clips_flatten[a_index[aa]])
            a_ch3.append(self.audio_ch3_clips_flatten[a_index[aa]])
            a_ch4.append(self.audio_ch4_clips_flatten[a_index[aa]])
        audios.append([torch.cat(a_ch1), torch.cat(a_ch2), torch.cat(a_ch3), torch.cat(a_ch4)])
        audios = torch.stack(audios[0], dim=0)
        audios = torch.mean(audios, dim=0).unsqueeze(0)  # ambisonics to mono

        return imgs, gts, audios, seq_name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

#dataloader for training
def get_loader(root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


#test dataset and loader
class test_dataset:
    def __init__(self, root, testsize):
        self.testsize = testsize

        with open(root, 'r') as f:
            self.seqs = [x.strip() for x in f.readlines()]

        video_clips, gt_clips = [], []
        audio_clips_ch1, audio_clips_ch2, audio_clips_ch3, audio_clips_ch4 = [], [], [], []
        for seq_idx in self.seqs:
            # get visual clips of each video
            frm_list = os.listdir(os.path.join('/home/yzhang1/PythonProjects/AV360/frame_key/', seq_idx))
            frm_list = sorted(frm_list)
            gt_list = os.listdir(os.path.join('/home/yzhang1/PythonProjects/AV360/DATA/test/', seq_idx))
            gt_list = sorted(gt_list)
            for idx in range(len(frm_list)):
                frm_list[idx] = '/home/yzhang1/PythonProjects/AV360/frame_key/' + seq_idx + '/' + frm_list[idx]
                gt_list[idx] = '/home/yzhang1/PythonProjects/AV360/DATA/test/' + seq_idx + '/' + gt_list[idx]
            frm_list_split = list(split_list(frm_list, int(len(frm_list) / clip_length)))
            gt_list_split = list(split_list(gt_list, int(len(gt_list) / clip_length)))
            video_clips.append(frm_list_split)
            gt_clips.append(gt_list_split)

            # get audio clips of each video
            audio_pth = '/home/yzhang1/PythonProjects/AV360/ambisonics_trimmed/' + seq_idx + '.wav'
            audio_ori = torchaudio.load(audio_pth)[0]
            audio_ori_ch1 = audio_ori[0]
            audio_ori_ch2 = audio_ori[1]
            audio_ori_ch3 = audio_ori[2]
            audio_ori_ch4 = audio_ori[3]
            audio_split_size = int(len(audio_ori[1]) / (int(len(frm_list) / clip_length)))
            audio_ch1_split = torch.split(tensor=audio_ori_ch1, split_size_or_sections=audio_split_size)
            audio_ch1_split = list(audio_ch1_split)
            audio_ch1_split = audio_ch1_split[:int(len(frm_list) / clip_length)]
            audio_clips_ch1.append(audio_ch1_split)
            audio_ch2_split = torch.split(tensor=audio_ori_ch2, split_size_or_sections=audio_split_size)
            audio_ch2_split = list(audio_ch2_split)
            audio_ch2_split = audio_ch2_split[:int(len(frm_list) / clip_length)]
            audio_clips_ch2.append(audio_ch2_split)
            audio_ch3_split = torch.split(tensor=audio_ori_ch3, split_size_or_sections=audio_split_size)
            audio_ch3_split = list(audio_ch3_split)
            audio_ch3_split = audio_ch3_split[:int(len(frm_list) / clip_length)]
            audio_clips_ch3.append(audio_ch3_split)
            audio_ch4_split = torch.split(tensor=audio_ori_ch4, split_size_or_sections=audio_split_size)
            audio_ch4_split = list(audio_ch4_split)
            audio_ch4_split = audio_ch4_split[:int(len(frm_list) / clip_length)]
            audio_clips_ch4.append(audio_ch4_split)

        self.video_clips_flatten = [item for sublist in video_clips for item in sublist]
        self.gt_clips_flatten = [item for sublist in gt_clips for item in sublist]

        self.audio_ch1_clips_flatten = [item for sublist in audio_clips_ch1 for item in sublist]
        self.audio_ch2_clips_flatten = [item for sublist in audio_clips_ch2 for item in sublist]
        self.audio_ch3_clips_flatten = [item for sublist in audio_clips_ch3 for item in sublist]
        self.audio_ch4_clips_flatten = [item for sublist in audio_clips_ch4 for item in sublist]

        # tools for data transform
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize * 2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform_er_cube = transforms.Compose([
            transforms.Resize((640, 1280)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.size = len(self.video_clips_flatten)
        self.index = 0

    def load_data(self):
        seq_name = self.video_clips_flatten[self.index][0].split('/')[-2]
        imgs, ER_imgs, gts, audios, frm_names = [], [], [], [], []
        for idx in range(clip_length):  # default as 3 consecutive frames of each clip
            curr_img = self.rgb_loader(self.video_clips_flatten[self.index][idx])
            ER_imgs.append(self.transform_er_cube(curr_img).unsqueeze(0))
            imgs.append(self.transform(curr_img).unsqueeze(0))
            curr_gt = self.binary_loader(self.gt_clips_flatten[self.index][idx])
            curr_gt = curr_gt.resize((self.testsize * 2, self.testsize))
            gts.append(curr_gt)
            frm_names.append(self.video_clips_flatten[self.index][idx].split('/')[-1])
        audios.append([self.audio_ch1_clips_flatten[self.index], self.audio_ch2_clips_flatten[self.index],
                       self.audio_ch3_clips_flatten[self.index], self.audio_ch4_clips_flatten[self.index]])
        audios = torch.stack(audios[0], dim=0).unsqueeze(0)

        self.index += 1
        self.index = self.index % self.size

        return imgs, ER_imgs, gts, audios, seq_name, frm_names

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


class dataset_inference:
    def __init__(self, root, testsize):
        self.testsize = testsize

        with open(root, 'r') as f:
            self.seqs = [x.strip() for x in f.readlines()]

        video_clips, gt_clips = [], []
        audio_clips_ch1, audio_clips_ch2, audio_clips_ch3, audio_clips_ch4 = [], [], [], []
        for seq_idx in self.seqs:
            # get visual clips of each video
            frm_list = os.listdir(os.path.join('/home/yzhang1/PythonProjects/AV360/frame_key/', seq_idx))
            frm_list = sorted(frm_list)
            gt_list = os.listdir(os.path.join('/home/yzhang1/PythonProjects/AV360/DATA/test/', seq_idx))
            gt_list = sorted(gt_list)
            for idx in range(len(frm_list)):
                frm_list[idx] = '/home/yzhang1/PythonProjects/AV360/frame_key/' + seq_idx + '/' + frm_list[idx]
                gt_list[idx] = '/home/yzhang1/PythonProjects/AV360/DATA/test/' + seq_idx + '/' + gt_list[idx]
            frm_list_split = list(split_list(frm_list, int(len(frm_list) / clip_length)))
            gt_list_split = list(split_list(gt_list, int(len(gt_list) / clip_length)))
            video_clips.append(frm_list_split)
            gt_clips.append(gt_list_split)

            # get audio clips of each video
            audio_pth = '/home/yzhang1/PythonProjects/AV360/ambisonics_trimmed/' + seq_idx + '.wav'
            audio_ori = torchaudio.load(audio_pth)[0]
            audio_ori_ch1 = audio_ori[0]
            audio_ori_ch2 = audio_ori[1]
            audio_ori_ch3 = audio_ori[2]
            audio_ori_ch4 = audio_ori[3]
            audio_split_size = int(len(audio_ori[1]) / (int(len(frm_list) / clip_length)))
            audio_ch1_split = torch.split(tensor=audio_ori_ch1, split_size_or_sections=audio_split_size)
            audio_ch1_split = list(audio_ch1_split)
            audio_ch1_split = audio_ch1_split[:int(len(frm_list) / clip_length)]
            audio_clips_ch1.append(audio_ch1_split)
            audio_ch2_split = torch.split(tensor=audio_ori_ch2, split_size_or_sections=audio_split_size)
            audio_ch2_split = list(audio_ch2_split)
            audio_ch2_split = audio_ch2_split[:int(len(frm_list) / clip_length)]
            audio_clips_ch2.append(audio_ch2_split)
            audio_ch3_split = torch.split(tensor=audio_ori_ch3, split_size_or_sections=audio_split_size)
            audio_ch3_split = list(audio_ch3_split)
            audio_ch3_split = audio_ch3_split[:int(len(frm_list) / clip_length)]
            audio_clips_ch3.append(audio_ch3_split)
            audio_ch4_split = torch.split(tensor=audio_ori_ch4, split_size_or_sections=audio_split_size)
            audio_ch4_split = list(audio_ch4_split)
            audio_ch4_split = audio_ch4_split[:int(len(frm_list) / clip_length)]
            audio_clips_ch4.append(audio_ch4_split)

        self.video_clips_flatten = [item for sublist in video_clips for item in sublist]
        self.gt_clips_flatten = [item for sublist in gt_clips for item in sublist]

        self.audio_ch1_clips_flatten = [item for sublist in audio_clips_ch1 for item in sublist]
        self.audio_ch2_clips_flatten = [item for sublist in audio_clips_ch2 for item in sublist]
        self.audio_ch3_clips_flatten = [item for sublist in audio_clips_ch3 for item in sublist]
        self.audio_ch4_clips_flatten = [item for sublist in audio_clips_ch4 for item in sublist]

        # tools for data transform
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize * 2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.size = len(self.video_clips_flatten)
        self.index = 0

    def load_data(self):
        seq_name = self.video_clips_flatten[self.index][0].split('/')[-2]
        imgs, gts, audios, frm_names = [], [], [], []
        for idx in range(clip_length):  # default as 3 consecutive frames of each clip
            curr_img = self.rgb_loader(self.video_clips_flatten[self.index][idx])
            imgs.append(self.transform(curr_img).unsqueeze(0))

            curr_gt = self.binary_loader(self.gt_clips_flatten[self.index][idx])
            gts.append(curr_gt)

            frm_names.append(self.video_clips_flatten[self.index][idx].split('/')[-1])

        # collect audio clip  (default as 15 frames equivalently; five times of clip length)
        if self.index == 0:
            a_index = [0, 1, 2, 3, 4]
        elif self.index == 1:
            a_index = [0, 1, 2, 3, 4]
        elif self.index == self.size - 1:
            a_index = [self.size - 5, self.size - 4, self.size - 3, self.size - 2, self.size - 1]
        elif self.index == self.size - 2:
            a_index = [self.size - 5, self.size - 4, self.size - 3, self.size - 2, self.size - 1]
        else:
            if self.video_clips_flatten[self.index - 2][0].split('/')[-2] != seq_name:
                if self.video_clips_flatten[self.index - 1][0].split('/')[-2] != seq_name:
                    a_index = [self.index, self.index + 1, self.index + 2, self.index + 3, self.index + 4]
                else:
                    a_index = [self.index - 1, self.index, self.index + 1, self.index + 2, self.index + 3]
            elif self.video_clips_flatten[self.index + 2][0].split('/')[-2] != seq_name:
                if self.video_clips_flatten[self.index + 1][0].split('/')[-2] != seq_name:
                    a_index = [self.index - 4, self.index - 3, self.index - 2, self.index - 1, self.index]
                else:
                    a_index = [self.index - 3, self.index - 2, self.index - 1, self.index, self.index + 1]
            else:
                a_index = [self.index - 2, self.index - 1, self.index, self.index + 1, self.index + 2]

        a_ch1, a_ch2, a_ch3, a_ch4 = [], [], [], []
        for aa in range(5):
            a_ch1.append(self.audio_ch1_clips_flatten[a_index[aa]])
            a_ch2.append(self.audio_ch2_clips_flatten[a_index[aa]])
            a_ch3.append(self.audio_ch3_clips_flatten[a_index[aa]])
            a_ch4.append(self.audio_ch4_clips_flatten[a_index[aa]])
        audios.append([torch.cat(a_ch1), torch.cat(a_ch2), torch.cat(a_ch3), torch.cat(a_ch4)])
        audios = torch.stack(audios[0], dim=0)
        audios = torch.mean(audios, dim=0).unsqueeze(0).unsqueeze(0)

        self.index += 1
        self.index = self.index % self.size

        return imgs, gts, audios, seq_name, frm_names

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
