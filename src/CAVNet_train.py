import os
import torch
import numpy as np
from datetime import datetime
from data import get_loader, test_dataset, clip_length
from utils import clip_gradient
import logging
import torch.backends.cudnn as cudnn
from options import opt
from utils import print_network, structure_loss, linear_annealing, l2_regularisation
import cv2

from models.CAVNet import cavnet


#set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

#build the model
model = cavnet()
print_network(model, 'CAVNet')
if(opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)
model.cuda()

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

#set the path
tr_root = opt.tr_root
te_root = opt.te_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(tr_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(te_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("CAVNet-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};load:{};save_path:{}'.
             format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.load,save_path))

step = 0
best_mae = 1
best_epoch = 0

#train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (imgs, gts, audios, seq_name) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            audios = audios.cuda()
            for idx in range(clip_length):  # default as three consecutive frames
                imgs[idx], gts[idx] = imgs[idx].cuda(), gts[idx].cuda()

            # debug
            # cv2.imwrite('img.png', imgs[1][0].permute(1, 2, 0).cpu().data.numpy() * 255)
            # cv2.imwrite('er_img.png', er_imgs[1][0].permute(1, 2, 0).cpu().data.numpy() * 255)
            # cv2.imwrite('gt.png', gts[1][0].permute(1, 2, 0).cpu().data.numpy() * 255)
            # cv2.imwrite('aem.png', aems[1][0].permute(1, 2, 0).cpu().data.numpy() * 255)
            # cv2.imwrite('cube1.png', cube_gts[1][0].permute(1, 2, 0).cpu().data.numpy() * 255)
            # cv2.imwrite('cube2.png', cube_gts[1][1].permute(1, 2, 0).cpu().data.numpy() * 255)
            # cv2.imwrite('cube3.png', cube_gts[1][2].permute(1, 2, 0).cpu().data.numpy() * 255)
            # cv2.imwrite('cube4.png', cube_gts[1][3].permute(1, 2, 0).cpu().data.numpy() * 255)
            # cv2.imwrite('cube5.png', cube_gts[1][4].permute(1, 2, 0).cpu().data.numpy() * 255)
            # cv2.imwrite('cube6.png', cube_gts[1][5].permute(1, 2, 0).cpu().data.numpy() * 255)
            # torchaudio.save('audio.wav', audios[0].cpu(), 48000)

            preds_prior, preds_post, lat_loss = model(imgs, audios, gts)

            anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
            reg_loss = (l2_regularisation(model.enc_mm_x) + l2_regularisation(model.enc_mm_xy) + \
                       l2_regularisation(model.backbone_dec_prior) + l2_regularisation(model.backbone_dec_post)) * 1e-4
            loss_list = []
            for rr in range(clip_length):  # prior
                loss_list.append(structure_loss(preds_prior[rr], gts[rr]))
            for pp in range(clip_length):  # post
                loss_list.append(structure_loss(preds_post[pp], gts[pp]))
            for cc in range(clip_length):  # latent loss
                loss_list.append(lat_loss[cc] * anneal_reg * opt.lat_weight)
            loss = sum(loss_list) / (clip_length * 3)
            loss = loss + reg_loss

            loss_vis_prior, loss_vis_post, loss_vis_lat = sum(loss_list[:3]) / clip_length, \
                                              sum(loss_list[3:6]) / clip_length, sum(loss_list[6:]) / clip_length

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 500 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, '
                      'LossPri: {:.4f}, LossPos: {:.4f}, LossLat: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data,
                           loss_vis_prior.data, loss_vis_post.data, loss_vis_lat.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, '
                             'LossPri: {:.4f}, LossPos: {:.4f}, LossLat: {:.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data,
                           loss_vis_prior.data, loss_vis_post.data, loss_vis_lat.data))
        
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path+'CAVNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'CAVNet_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise

 
if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        train(train_loader, model, optimizer, epoch, save_path)
