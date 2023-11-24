import os
import PIL.Image as Image
emo2label = {'disgust':0,'fear':0,'sad':0,'anger':0,'happy':1,'surprise':2,'others':3,'Others':3 ,'positive':1,'negative':0}

import torch
import torchvision.transforms.functional as f





def get_test_data(root= './data/v2_micro_cut'):

    """ root = './data/v2_micro_cut' """

    files = []
    labels = []
    for sub in os.listdir(root):
        for emo in os.listdir(os.path.join(root, sub)):
            for fn in os.listdir(os.path.join(root, sub, emo)):
                imgs = os.listdir(os.path.join(root, sub, emo, fn))
                # print(len(imgs),os.path.join(root,sub,emo,fn))
                if len(imgs) == 2:
                    if emo2label[emo] != 3:
                        if 'v2' in root:
                            apex = os.path.join(root, sub, emo, fn, imgs[0])
                            onset = os.path.join(root, sub, emo, fn, imgs[1])
                            files.append([onset, apex])
                            labels.append(emo2label[emo])

                        else:
                            onset = os.path.join(root, sub, emo, fn, imgs[0])
                            apex = os.path.join(root, sub, emo, fn, imgs[1])
                            files.append([onset, apex])
                            labels.append(emo2label[emo])
                    # print(path_0,path_1,emo2label[emo])
    # print(len(files),val)
    return files, labels


def get_data_aug():

    files = []
    root = './MIX_Augment/CASME2'
    subs = os.listdir(root)
    for sub in subs:
        emos = os.listdir(os.path.join(root,sub))
        for emo in emos:
            fns = os.listdir(os.path.join(root,sub,emo))
            for fn in fns:
                augs = os.listdir(os.path.join(root,sub,emo,fn))
                for aug in augs:
                    imgs = os.listdir(os.path.join(root,sub,emo,fn,aug))
                    onset = os.path.join(root,sub,emo,fn,aug,imgs[0])
                    apex = os.path.join(root, sub, emo, fn, aug,imgs[1])

                    files.append([onset,apex])
    print('CASME2 ALL READY')
    root = './MIX_Augment/v2_macro'
    subs = os.listdir(root)
    for sub in subs:
        fns = os.listdir(os.path.join(root,sub))
        for fn in fns:
            augs = os.listdir(os.path.join(root,sub,fn))
            for aug in augs:
                imgs = os.listdir(os.path.join(root,sub,fn,aug))
                if imgs:
                    sec = os.path.join(root, sub, fn, aug,imgs[0])
                    onset = os.path.join(root, sub,  fn, aug,imgs[1])
                    files.append([onset, sec])

    print('CASME3 MACRO ALL READY')

    return files


class train_dataset(torch.utils.data.Dataset):
    def __init__(self,transform):
        super(train_dataset, self).__init__()
        self.files = get_data_aug()
        self.transform = transform
        # self.rotate = torchvision.transforms.functional.rotate()
    def __getitem__(self, item,add_rotute = True):
        onset = self.transform(Image.open(self.files[item][0]))
        apex = self.transform(Image.open(self.files[item][1]))
        if add_rotute:
            rotate = f.rotate(self.transform(Image.open(self.files[item][1])),90)
            return [onset,apex,rotate]
        return [onset,apex]
    def __len__(self):
        return len(self.files)





class test_dataset(torch.utils.data.Dataset):
    def __init__(self,root,transform):
        '''

        :param root:= './data/v2_micro_cut'
        '''

        super(test_dataset, self).__init__()
        self.files ,self.labels = get_test_data(root)
        self.transform = transform

    def __getitem__(self, item):
        onset = self.transform(Image.open(self.files[item][0]))
        apex = self.transform(Image.open(self.files[item][1]))
        rotate = f.rotate(self.transform(Image.open(self.files[item][1])), 90)
        # print(self.files[item])
        return [onset,apex,rotate]

    def __len__(self):
        return len(self.files)

