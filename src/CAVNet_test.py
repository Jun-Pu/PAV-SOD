import torch
import numpy as np
import os, argparse
import cv2
from data import dataset_inference, clip_length

from models.CAVNet import cavnet

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default=os.getcwd() + '/data/', help='test dataset path')
parser.add_argument('--forward_iter', type=int, default=5, help='sample times')
opt = parser.parse_args()

dataset_path = opt.test_path

# set device for test
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

# load the model
model = cavnet()
model.load_state_dict(torch.load(os.getcwd() + '/model_test/CAVNet_final.pth'))
model.cuda()
model.eval()

#test
TIME = []
save_path_pred = './predictions/'
save_path_sigma = './uncertainties/'
if not os.path.exists(save_path_pred): os.makedirs(save_path_pred)
if not os.path.exists(save_path_sigma): os.makedirs(save_path_sigma)
root = os.path.join(dataset_path, 'PAVS10K_seqs_test.txt')
test_loader = dataset_inference(root, opt.testsize)
for i in range(test_loader.size):
    imgs, gts, audios, seq_name, frm_names = test_loader.load_data()
    for idx in range(clip_length):
        gts[idx] = np.asarray(gts[idx], np.float32)
        gts[idx] /= (gts[idx].max() + 1e-8)
        imgs[idx] = imgs[idx].cuda()
    audios = audios.cuda()

    # multiple forward
    pred1, pred2, pred3, ent1, ent2, ent3 = [], [], [], [], [], []
    with torch.no_grad():
        for ff in range(opt.forward_iter):
            pred_curr = model(imgs, audios)
            p1, p2, p3 = torch.sigmoid(pred_curr[0]), torch.sigmoid(pred_curr[1]), torch.sigmoid(pred_curr[2])

            pred1.append(p1)
            pred2.append(p2)
            pred3.append(p3)
            ent1.append(-1 * p1 * torch.log(p1 + 1e-8))
            ent2.append(-1 * p2 * torch.log(p2 + 1e-8))
            ent3.append(-1 * p3 * torch.log(p3 + 1e-8))

    pred1_c, pred2_c, pred3_c = torch.cat(pred1, dim=1), torch.cat(pred2, dim=1), torch.cat(pred3, dim=1)
    pred1_mu, pred2_mu, pred3_mu = torch.mean(pred1_c, 1, keepdim=True), torch.mean(pred2_c, 1, keepdim=True), \
                                   torch.mean(pred3_c, 1, keepdim=True)
    ent1_c, ent2_c, ent3_c = torch.cat(ent1, dim=1), torch.cat(ent2, dim=1), torch.cat(ent3, dim=1)
    ent1_mu, ent2_mu, ent3_mu = torch.mean(ent1_c, 1, keepdim=True), torch.mean(ent2_c, 1, keepdim=True), \
                                   torch.mean(ent3_c, 1, keepdim=True)

    # before save results
    pred1_mu, pred2_mu, pred3_mu = pred1_mu.data.cpu().numpy().squeeze(), pred2_mu.data.cpu().numpy().squeeze(),\
                                   pred3_mu.data.cpu().numpy().squeeze()
    ent1_mu, ent2_mu, ent3_mu = ent1_mu.data.cpu().numpy().squeeze(), ent2_mu.data.cpu().numpy().squeeze(), \
                                   ent3_mu.data.cpu().numpy().squeeze()

    pred1_mu = (pred1_mu - pred1_mu.min()) / (pred1_mu.max() - pred1_mu.min() + 1e-8)
    pred2_mu = (pred2_mu - pred2_mu.min()) / (pred2_mu.max() - pred2_mu.min() + 1e-8)
    pred3_mu = (pred3_mu - pred3_mu.min()) / (pred3_mu.max() - pred3_mu.min() + 1e-8)
    ent1_mu = 255 * (ent1_mu - ent1_mu.min()) / (ent1_mu.max() - ent1_mu.min() + 1e-8)
    ent2_mu = 255 * (ent2_mu - ent2_mu.min()) / (ent2_mu.max() - ent2_mu.min() + 1e-8)
    ent3_mu = 255 * (ent3_mu - ent3_mu.min()) / (ent3_mu.max() - ent3_mu.min() + 1e-8)

    uc1, uc2, uc3 = ent1_mu.astype(np.uint8), ent2_mu.astype(np.uint8), ent3_mu.astype(np.uint8)
    uc1, uc2, uc3 = cv2.applyColorMap(uc1, cv2.COLORMAP_JET), cv2.applyColorMap(uc2, cv2.COLORMAP_JET), \
                    cv2.applyColorMap(uc3, cv2.COLORMAP_JET)

    # save results
    curr_save_pth_pred = os.path.join(save_path_pred, seq_name)
    curr_save_pth_sigma = os.path.join(save_path_sigma, seq_name)
    if not os.path.exists(curr_save_pth_pred): os.makedirs(curr_save_pth_pred)
    if not os.path.exists(curr_save_pth_sigma): os.makedirs(curr_save_pth_sigma)

    print('save prediction to: ', os.path.join(curr_save_pth_pred, frm_names[0]))
    cv2.imwrite(os.path.join(curr_save_pth_pred, frm_names[0]), pred1_mu * 255)
    print('save prediction to: ', os.path.join(curr_save_pth_pred, frm_names[1]))
    cv2.imwrite(os.path.join(curr_save_pth_pred, frm_names[1]), pred2_mu * 255)
    print('save prediction to: ', os.path.join(curr_save_pth_pred, frm_names[2]))
    cv2.imwrite(os.path.join(curr_save_pth_pred, frm_names[2]), pred3_mu * 255)

    print('save uncertainty to: ', os.path.join(curr_save_pth_sigma, frm_names[0]))
    cv2.imwrite(os.path.join(curr_save_pth_sigma, frm_names[0]), uc1)
    print('save uncertainty to: ', os.path.join(curr_save_pth_sigma, frm_names[1]))
    cv2.imwrite(os.path.join(curr_save_pth_sigma, frm_names[1]), uc2)
    print('save uncertainty to: ', os.path.join(curr_save_pth_sigma, frm_names[2]))
    cv2.imwrite(os.path.join(curr_save_pth_sigma, frm_names[2]), uc3)

print('Test Done!')
