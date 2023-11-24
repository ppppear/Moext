import torchvision.transforms as transforms
import torch
import os
import sys
from torch.utils import data


from data_split import train_dataset,test_dataset
from train_base import train_base
from MDnet.unet_ca import net


root = '../data/MIX_aug'
test_root = '../data/MIX'

param_path = '../param_motion/unet_Res_29_mine.pth'

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

subs = os.listdir(root)

len = len(subs)
epoch = 30
list_casme = [[] for j in range(3)]
list_samm = [[] for j in range(3)]
list_smic = [[] for j in range(3)]


uar, uf1, acc = 0, 0, 0
for i in range(len):

    val = subs[i]
    print('============================     sub:{}     val:{}     ============================'.format(i,val))
    train_data = train_dataset(root,val,transform,True)
    test_data = test_dataset(test_root,val,transform)

    train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=3)
    test_loader = data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=3)
    Net = net()

    train_processing = train_base(epoch,train_loader,test_loader,Net,param_path)
    train_processing.train_processing()



    acc += train_processing.acc
    uf1 += train_processing.uf1
    uar += train_processing.uar
    i+=1
    print('================  sub:{}  acc:{:.4f}   uf1{:.4f}   uar{:.4f}  ====================='.format(i,acc/i,uf1/i,uar/i))
    if(val[0]== "0" ):
        list_smic[0].append(train_processing.acc)
        list_smic[1].append(train_processing.uf1)
        list_smic[2].append(train_processing.uar)


    elif(val[0]=="s"):
        list_samm[0].append(train_processing.acc)
        list_samm[1].append(train_processing.uf1)
        list_samm[2].append(train_processing.uar)

    else:
        list_casme[0].append(train_processing.acc)
        list_casme[1].append(train_processing.uf1)
        list_casme[2].append(train_processing.uar)

    print(list_casme)

print("casme:acc:{:.4f}   uf1{:.4f}   uar{:.4f}".format(sum(list_casme[0])/25,sum(list_casme[1])/25,sum(list_casme[2])/25))
print("samm:acc:{:.4f}   uf1{:.4f}   uar{:.4f}".format(sum(list_samm[0])/16,sum(list_samm[1])/16,sum(list_samm[2])/16))
print("smic:acc:{:.4f}   uf1{:.4f}   uar{:.4f}".format(sum(list_smic[0]) / 28, sum(list_smic[1])/28, sum(list_smic[2])/28))

f = open("res.txt","a")

f.write("casme:acc:{:.4f}   uf1{:.4f}   uar{:.4f}".format(sum(list_casme[0])/25,sum(list_casme[1])/25,sum(list_casme[2])/25))
f.write("samm:acc:{:.4f}   uf1{:.4f}   uar{:.4f}".format(sum(list_samm[0])/16,sum(list_samm[1])/16,sum(list_samm[2])/16))
f.write("smic:acc:{:.4f}   uf1{:.4f}   uar{:.4f}".format(sum(list_smic[0])/28, sum(list_smic[1])/28, sum(list_smic[2])/28))
