import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=21, help='epoch number')
parser.add_argument('--lr', type=float, default=2.5e-6, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')  # set as 1
parser.add_argument('--trainsize', type=int, default=416, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--lat_weight', type=int, default=10, help='weighting latent loss')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--tr_root', type=str, default=os.getcwd() + '/data/PAVS10K_seqs_train.txt', help='')
parser.add_argument('--te_root', type=str, default=os.getcwd() + '/data/PAVS10K_seqs_test.txt', help='')
parser.add_argument('--save_path', type=str, default=os.getcwd() + '/CAVNet_cpts/', help='the path to save models and logs')
opt = parser.parse_args()
