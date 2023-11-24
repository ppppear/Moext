from math import exp

import torch
import os

import torch.nn as nn
from torch.utils import data
from torchvision import transforms

from data_split import *
from unet_base import Motion_net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

print(torch.cuda.device_count())
print(torch.cuda.is_available())

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])




train_data = train_dataset(transform)
test_data = test_dataset('./data/v2_micro_cut',transform)




train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=3)
test_loader = data.DataLoader(test_data, batch_size=32 ,shuffle=True, num_workers=3)


len_t = len(train_loader)
len_v = len(test_loader)


frist_net = Motion_net().cuda()

pretrained_state = torch.load('./param_motion/unet_Res_29_mine.pth')
model_dict = frist_net.state_dict()
model_dict.update(pretrained_state)
frist_net.load_state_dict(model_dict)

optimizer = torch.optim.Adam(frist_net.parameters(), lr=0.0001, weight_decay=1e-4)
epoch = 15



for i in range(epoch):
    # print('00000')
    v_loss, v_loss1, v_loss2, v_loss3,v_loss4,v_loss5 = 0, 0, 0, 0,0,0
    loss, t_loss, loss1, loss2, loss3 ,loss4,loss5= 0, 0, 0, 0, 0,0,0
    j = 0

    for imgs in train_loader:

        frist_net.train()
        # print(j)
        y_hat, texture_BC, texture_AB, motion_BC, mo_tex = frist_net(imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda())
        criterion = nn.MSELoss().cuda()
        loss_l1 = criterion(y_hat, imgs[1].cuda())
        loss_mot_tex = torch.exp(-criterion(*texture_BC))
        loss_texAB = criterion(*texture_AB)
        loss_motBC = 0.5*criterion(*motion_BC)
        loss_motAC = torch.exp(-criterion(*mo_tex))
        loss = loss_l1 + loss_mot_tex + loss_texAB + loss_motBC + loss_motAC
        # print(loss_l1,loss_mot_tex,loss_motAC)
        # loss = loss_l1
        t_loss += loss
        loss1 += loss_l1
        loss2 += loss_mot_tex
        loss3 += loss_texAB
        loss4 += loss_motBC
        loss5 += loss_motAC

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        j += 1

        if (j + 1) % 30 == 0:
            print(
                '       j:{} loss  {:.4F} loss_l1  {:.4F} loss_mot_tex  {:.4F} loss_texAB: {:.4F} loss_motBC: {:.4F} loss_motAC: {:.4F}'
                    .format((j+1)/30, t_loss / j, loss1 / j, loss2 / j ,loss3 / j,loss4/j,loss5/j))

    with torch.no_grad():
        for imgs in test_loader:
            frist_net.eval()
            # print(imgs)

            y_hat, texture_BC, texture_AB, motion_BC, mo_tex = frist_net(imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda())
            criterion = nn.MSELoss().cuda()
            loss_l1 = criterion(y_hat, imgs[1].cuda())
            loss_mot_tex = torch.exp(-criterion(*texture_BC))
            loss_texAB = criterion(*texture_AB)
            loss_motBC = 0.5 * criterion(*motion_BC)
            loss_motAC = torch.exp(-criterion(*mo_tex))
            loss = loss_l1 + loss_mot_tex + loss_texAB + loss_motBC + loss_motAC

            v_loss += loss
            v_loss1 += loss_l1
            v_loss2 += loss_mot_tex
            v_loss3 += loss_texAB
            v_loss4 += loss_motBC
            v_loss5 += loss_motAC



    print(
        'epoch: {} loss: {:.4F} loss_l1: {:.4F} loss_mot_tex: {:.4F} loss_texAB: {:.4F} loss_motBC: {:.4F} loss_motAC: {:.4F}\n ============================================ loss{:.4F} loss_l1{:.4F} loss_mot_tex{:.4F} loss_texAB: {:.4F}  loss_motBC: {:.4F} loss_motAC: {:.4F}'
        .format(i, t_loss / len_t, loss1 / len_t, loss2 / len_t, loss3 / len_t,loss4/len_t,loss5/len_t , v_loss / len_v, v_loss1 / len_v,
                v_loss2 / len_v,
                v_loss3 / len_v,v_loss4/len_v,v_loss5/len_v))

    if  v_loss / len_v<1.1:
        torch.save(frist_net.state_dict(),os.path.join('./param_motion','unet_Res_'+str(i+30)+'_mine.pth'))
