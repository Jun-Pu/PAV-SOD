import os
import cv2

from models.rcrnet_vit import RCRNet_vit, _ConvBatchNormReLU, _RefinementModule
from models.blocks import forward_vit
from models.convgru import ConvGRUCell
from models.non_local_dot_product import NONLocalBlock3D

from collections import OrderedDict
from torch.autograd import Variable
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Independent, kl


class InferenceModel_mm_x(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(InferenceModel_mm_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1 = nn.Linear(channels * 8 * 13 * 26, latent_size)  # adjust according to input size
        self.fc2 = nn.Linear(channels * 8 * 13 * 26, latent_size)  # adjust according to input size

        self.a_fc1 = nn.Linear(1024, latent_size)
        self.a_fc2 = nn.Linear(1024, latent_size)

        self.av_fc1 = nn.Bilinear(latent_size, latent_size, latent_size)
        self.av_fc2 = nn.Bilinear(latent_size, latent_size, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input, aux_input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        output = output.view(-1, self.channel * 8 * 13 * 26)  # adjust according to input size
        # print(output.size())
        # output = self.tanh(output)

        # audio visual fusion
        mu = self.fc1(output)
        a_mu = self.a_fc1(aux_input)
        av_mu = self.av_fc1(mu, a_mu)

        logvar = self.fc2(output)
        a_logvar = self.a_fc2(aux_input)
        av_logvar = self.av_fc2(logvar, a_logvar)
        dist = Independent(Normal(loc=av_mu, scale=torch.exp(av_logvar)), 1)

        return av_mu, av_logvar, dist


class InferenceModel_mm_xy(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(InferenceModel_mm_xy, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1 = nn.Linear(channels * 8 * 13 * 26, latent_size)  # adjust according to input size
        self.fc2 = nn.Linear(channels * 8 * 13 * 26, latent_size)  # adjust according to input size

        self.a_fc1 = nn.Linear(1024, latent_size)
        self.a_fc2 = nn.Linear(1024, latent_size)

        self.av_fc1 = nn.Bilinear(latent_size, latent_size, latent_size)
        self.av_fc2 = nn.Bilinear(latent_size, latent_size, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input, aux_input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        output = output.view(-1, self.channel * 8 * 13 * 26)  # adjust according to input size
        # print(output.size())
        # output = self.tanh(output)

        # audio visual fusion
        mu = self.fc1(output)
        a_mu = self.a_fc1(aux_input)
        av_mu = self.av_fc1(mu, a_mu)

        logvar = self.fc2(output)
        a_logvar = self.a_fc2(aux_input)
        av_logvar = self.av_fc2(logvar, a_logvar)
        dist = Independent(Normal(loc=av_mu, scale=torch.exp(av_logvar)), 1)

        return av_mu, av_logvar, dist


class cavnet(nn.Module):
    def __init__(self, output_stride=16, lat_channel=16, lat_dim=32):
        super(cavnet, self).__init__()
        # for visual backbone
        # encoder
        self.backbone_enc_vit = RCRNet_vit(n_classes=1, output_stride=output_stride).pretrained
        self.backbone_enc_aspp = RCRNet_vit(n_classes=1, output_stride=output_stride).aspp

        # decoder
        self.backbone_dec_prior = RCRNet_decoder()
        self.backbone_dec_post = RCRNet_decoder()

        # for temporal module
        self.convgru_forward = ConvGRUCell(256, 256, 3)
        self.convgru_backward = ConvGRUCell(256, 256, 3)
        self.bidirection_conv = nn.Conv2d(512, 256, 3, 1, 1)
        self.non_local_block = NONLocalBlock3D(256, sub_sample=False, bn_layer=False)
        self.non_local_block2 = NONLocalBlock3D(256, sub_sample=False, bn_layer=False)

        # for CVAE
        self.enc_mm_x = InferenceModel_mm_x(3, lat_channel, lat_dim)
        self.enc_mm_xy = InferenceModel_mm_xy(4, lat_channel, lat_dim)
        self.spatial_axes = [2, 3]
        self.noise_conv_prior = nn.Conv2d(256 + lat_dim, 256, kernel_size=1, padding=0)
        self.noise_conv_post = nn.Conv2d(256 + lat_dim, 256, kernel_size=1, padding=0)

        # for audio
        # enocder
        self.soundnet = nn.Sequential(  # 7 layers used
            nn.Conv2d(1, 16, (1, 64), (1, 2), (0, 32)), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((1, 8), (1, 8)),
            nn.Conv2d(16, 32, (1, 32), (1, 2), (0, 16)), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1, 8), (1, 8)),
            nn.Conv2d(32, 64, (1, 16), (1, 2), (0, 8)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, (1, 8), (1, 2), (0, 4)), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, (1, 4), (1, 2), (0, 2)), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((1, 4), (1, 4)),
            nn.Conv2d(256, 512, (1, 4), (1, 2), (0, 2)), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 1024, (1, 4), (1, 2), (0, 2)), nn.BatchNorm2d(1024), nn.ReLU(), nn.MaxPool2d((1, 2))
        )

        # load pre-training model
        if self.training:
            self.initialize_pretrain()  # load visual pretrain (static visual part pre-trained on DUTS-tr)
            self.initialize_soundnet()  # load audio pretrain (soundnet)

        # may freeze auditory branch
        for param in self.soundnet.parameters(): param.requires_grad = True

    def initialize_pretrain(self):
        backbone_pretrain = torch.load(os.getcwd() + '/pretrain/static_visual_pretrain.pth')

        all_params_enc_vit = {}
        for k, v in self.backbone_enc_vit.state_dict().items():
            if 'pretrained.' + k in backbone_pretrain.keys():
                v = backbone_pretrain['pretrained.' + k]
                all_params_enc_vit[k] = v
        self.backbone_enc_vit.load_state_dict(all_params_enc_vit)

        all_params_enc_aspp = {}
        for k, v in self.backbone_enc_aspp.state_dict().items():
            if 'aspp.' + k in backbone_pretrain.keys():
                v = backbone_pretrain['aspp.' + k]
                all_params_enc_aspp[k] = v
        self.backbone_enc_aspp.load_state_dict(all_params_enc_aspp)

    def initialize_soundnet(self):
        audio_pretrain_weights = torch.load(os.getcwd() + '/pretrain/soundnet8.pth')

        all_params = {}
        for k, v in self.soundnet.state_dict().items():
            if 'module.soundnet8.' + k in audio_pretrain_weights.keys():
                v = audio_pretrain_weights['module.soundnet8.' + k]
                all_params[k] = v
        self.soundnet.load_state_dict(all_params)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)

        return kl_div

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            device)
        return torch.index_select(a, dim, order_index)

    def forward(self, seq, audio_clip, gt=None):
        # --------------------------------------------------------------------------------------------------------------
        # encoder for mono sound
        feats_audio = self.soundnet(audio_clip.unsqueeze(0))
        feats_audio = torch.mean(feats_audio, dim=-1).squeeze().unsqueeze(0)

        # --------------------------------------------------------------------------------------------------------------
        # encoder for audio-visual-temporal
        feats_seq = [forward_vit(self.backbone_enc_vit, frame) for frame in seq]
        feats_time = [self.backbone_enc_aspp(feats[-1]) for feats in feats_seq]  # bottleneck features
        feats_time = torch.stack(feats_time, dim=2)
        feats_time = self.non_local_block(feats_time)

        # Deep Bidirectional ConvGRU
        frame = seq[0]
        feat = feats_time[:, :, 0, :, :]
        feats_forward = []
        # forward
        for i in range(len(seq)):
            feat = self.convgru_forward(feats_time[:, :, i, :, :], feat)
            feats_forward.append(feat)
        # backward
        feat = feats_forward[-1]
        feats_backward = []
        for i in range(len(seq)):
            feat = self.convgru_backward(feats_forward[len(seq)-1-i], feat)
            feats_backward.append(feat)

        feats_backward = feats_backward[::-1]
        feats = []
        for i in range(len(seq)):
            feat = torch.tanh(self.bidirection_conv(torch.cat((feats_forward[i], feats_backward[i]), dim=1)))
            feats.append(feat)
        feats = torch.stack(feats, dim=2)

        feats = self.non_local_block2(feats)  # spatial-temporal bottleneck features

        # --------------------------------------------------------------------------------------------------------------
        if gt == None:
            # model inference
            feats_prior = []
            for rr in range(len(seq)):
                mu_prior, logvar_prior, _ = self.enc_mm_x(seq[rr], feats_audio)
                z_prior = self.reparametrize(mu_prior, logvar_prior)  # instantiate latent variable
                z_prior = torch.unsqueeze(z_prior, 2)
                z_prior = self.tile(z_prior, 2, feats[:, :, rr, :, :].shape[self.spatial_axes[0]])
                z_prior = torch.unsqueeze(z_prior, 3)
                z_prior = self.tile(z_prior, 3, feats[:, :, rr, :, :].shape[self.spatial_axes[1]])
                f_prior = torch.cat((feats[:, :, rr, :, :], z_prior), 1)
                feats_prior.append(self.noise_conv_prior(f_prior))

            preds_prior = []
            for i, frame in enumerate(seq):
                seg_prior = self.backbone_dec_prior(feats_seq[i][0], feats_seq[i][1], feats_seq[i][2], feats_prior[i],
                                                    frame)
                preds_prior.append(seg_prior)

            return preds_prior
        else:
            # CVAE
            KLD, feats_prior, feats_post = [], [], []
            for rr in range(len(seq)):
                mu_prior, logvar_prior, dist_prior = self.enc_mm_x(seq[rr], feats_audio)
                z_prior = self.reparametrize(mu_prior, logvar_prior)  # instantiate latent variable
                z_prior = torch.unsqueeze(z_prior, 2)
                z_prior = self.tile(z_prior, 2, feats[:, :, rr, :, :].shape[self.spatial_axes[0]])
                z_prior = torch.unsqueeze(z_prior, 3)
                z_prior = self.tile(z_prior, 3, feats[:, :, rr, :, :].shape[self.spatial_axes[1]])
                f_prior = torch.cat((feats[:, :, rr, :, :], z_prior), 1)
                feats_prior.append(self.noise_conv_prior(f_prior))

                mu_post, logvar_post, dist_post = self.enc_mm_xy(torch.cat((seq[rr], gt[rr]), 1), feats_audio)
                z_post = self.reparametrize(mu_post, logvar_post)  # instantiate latent variable
                z_post = torch.unsqueeze(z_post, 2)
                z_post = self.tile(z_post, 2, feats[:, :, rr, :, :].shape[self.spatial_axes[0]])
                z_post = torch.unsqueeze(z_post, 3)
                z_post = self.tile(z_post, 3, feats[:, :, rr, :, :].shape[self.spatial_axes[1]])
                f_post = torch.cat((feats[:, :, rr, :, :], z_post), 1)
                feats_post.append(self.noise_conv_post(f_post))

                KLD.append(torch.mean(self.kl_divergence(dist_post, dist_prior)))

            preds_prior, preds_post = [], []
            for i, frame in enumerate(seq):
                seg_prior = self.backbone_dec_prior(feats_seq[i][0], feats_seq[i][1], feats_seq[i][2], feats_prior[i],
                                                    frame)
                preds_prior.append(seg_prior)
                seg_post = self.backbone_dec_post(feats_seq[i][0], feats_seq[i][1], feats_seq[i][2], feats_post[i],
                                            frame)
                preds_post.append(seg_post)

            return preds_prior, preds_post, KLD


class RCRNet_decoder(nn.Module):
    def __init__(self, n_classes=1):
        super(RCRNet_decoder, self).__init__()
        self.decoder = nn.Sequential(
                    OrderedDict(
                        [
                            ("conv1", _ConvBatchNormReLU(128, 256, 3, 1, 1, 1)),
                            ("conv2", nn.Conv2d(256, n_classes, kernel_size=1)),
                        ]
                    )
                )
        self.add_module("refinement1", _RefinementModule(768, 96, 256, 128, 2))
        self.add_module("refinement2", _RefinementModule(512, 96, 128, 128, 2))
        self.add_module("refinement3", _RefinementModule(256, 96, 128, 128, 2))

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if self.training: self.init_pretrain()

    def init_pretrain(self):
        backbone_pretrain = torch.load(os.getcwd() + '/pretrain/static_visual_pretrain.pth')

        all_params_dec = {}
        for k, v in self.decoder.state_dict().items():
            if 'decoder.' + k in backbone_pretrain.keys():
                v = backbone_pretrain['decoder.' + k]
                all_params_dec[k] = v
        self.decoder.load_state_dict(all_params_dec)

        all_params_ref1 = {}
        for k, v in self.refinement1.state_dict().items():
            if 'refinement1.' + k in backbone_pretrain.keys():
                v = backbone_pretrain['refinement1.' + k]
                all_params_ref1[k] = v
        self.refinement1.load_state_dict(all_params_ref1)

        all_params_ref2 = {}
        for k, v in self.refinement2.state_dict().items():
            if 'refinement2.' + k in backbone_pretrain.keys():
                v = backbone_pretrain['refinement2.' + k]
                all_params_ref2[k] = v
        self.refinement2.load_state_dict(all_params_ref2)

        all_params_ref3 = {}
        for k, v in self.refinement3.state_dict().items():
            if 'refinement3.' + k in backbone_pretrain.keys():
                v = backbone_pretrain['refinement3.' + k]
                all_params_ref3[k] = v
        self.refinement3.load_state_dict(all_params_ref3)

    def seg_conv(self, block1, block2, block3, block4, shape):
        '''
            Pixel-wise classifer
        '''
        block4 = self.upsample2(block4)

        bu1 = self.refinement1(block3, block4)
        bu1 = F.interpolate(bu1, size=block2.shape[2:], mode="bilinear", align_corners=False)
        bu2 = self.refinement2(block2, bu1)
        bu2 = F.interpolate(bu2, size=block1.shape[2:], mode="bilinear", align_corners=False)
        bu3 = self.refinement3(block1, bu2)
        bu3 = F.interpolate(bu3, size=shape, mode="bilinear", align_corners=False)
        seg = self.decoder(bu3)

        return seg

    def forward(self, block1, block2, block3, block4, x):
        seg = self.seg_conv(block1, block2, block3, block4, x.shape[2:])

        return seg

