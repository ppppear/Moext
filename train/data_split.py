import torch
import torch.nn
import os
import PIL.Image as Image
emo2label = {'disgust':0,'fear':0,'sad':0,'anger':0,'happy':1,'surprise':2,'others':3,'Others':3 ,'positive':1,'negative':0}


def get_train_data(root,val):


    files = []
    labels = []
    for sub in os.listdir(root):
        if sub!= val:
            for emo in os.listdir(os.path.join(root,sub)):
                fns = os.listdir(os.path.join(root, sub, emo))
                for fn in fns:
                    nums = os.listdir(os.path.join(root, sub, emo, fn))
                    for num in nums:
                        imgs = os.listdir(os.path.join(root, sub, emo, fn, num))

                        if len(imgs) == 2:
                            if emo2label[emo] != 3:

                                if 'v2' in root:
                                    apex = os.path.join(root, sub, emo, fn, num, imgs[0])
                                    onset = os.path.join(root, sub, emo, fn, num, imgs[1])
                                    files.append([onset, apex])
                                    labels.append(emo2label[emo])

                                else:

                                    onset = os.path.join(root, sub, emo, fn, num, imgs[0])
                                    apex = os.path.join(root, sub, emo, fn, num, imgs[1])
                                    files.append([onset, apex])
                                    labels.append(emo2label[emo])

    return files,labels


def get_test_data(root,val,test = True):
    files = []
    labels = []
    for sub in os.listdir(root):
        if test == True:
            if sub == val:
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
        else:
            if sub != val:
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

    return files, labels


class train_dataset(torch.utils.data.Dataset):
    def __init__(self,root,val,transform,aug = True):
        '''

        :param root:'./MIX_aug OR ./v2_aug'
        :param val: the filename of the sub used to val
        '''

        super(train_dataset, self).__init__()

        if aug :

            self.files ,self.labels = get_train_data(root,val)
        else:
            self.files, self.labels = get_test_data(root, val,False)
        self.transform = transform
        self.aug = aug
    def __getitem__(self, item):

        onset = self.transform(Image.open(self.files[item][0]))
        apex = self.transform(Image.open(self.files[item][1]))

        # print(self.labels[item])
        return [onset,apex],self.labels[item]

    def __len__(self):
        # print(len(self.files))
        return len(self.files)


class test_dataset(torch.utils.data.Dataset):
    def __init__(self,root,val,transform):
        '''

        :param root:'./MIX OR ./v2_micro_cut'
        :param val: the filename of the sub used to val
        '''

        super(test_dataset, self).__init__()
        self.files ,self.labels = get_test_data(root,val)
        self.transform = transform

    def __getitem__(self, item):
        onset = self.transform(Image.open(self.files[item][0]))
        apex = self.transform(Image.open(self.files[item][1]))

        # print(self.labels[item])
        return [onset,apex],self.labels[item]

    def __len__(self):
        return len(self.files)