import os
from sklearn import metrics
from torch import nn
import torch
from sklearn.metrics import confusion_matrix
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def confusionMatrix(gt, pred):
    TN, FP, FN, TP = confusion_matrix(gt, pred,labels=[0,1]).ravel()
    num_samples = len([x for x in gt if x == 1])
    if num_samples != 0:
        f1_score = (2 * TP) / (2 * TP + FP + FN)
        average_recall = TP / num_samples
        return f1_score, average_recall, num_samples
    else:
        return 0, 0, 0


def score2category(preds):
    category = []
    for i in range(len(preds)):
        category.append(torch.argmax(preds[i]))
    return category


def accuary(preds, labels):
    sum = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            sum += 1
    return sum / len(preds)


def recognition_evaluation(labels, preds, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}

    # Display recognition result
    uf1_list = []
    uar_list = []
    try:

        for emotion, emotion_index in label_dict.items():
            real_label = [1 if x == emotion_index else 0 for x in labels]
            pred_result = [1 if x == emotion_index else 0 for x in preds]
            try:

                f1_recog, ar_recog, tag = confusionMatrix(real_label, pred_result)
                if tag != 0:
                    uf1_list.append(f1_recog)
                    uar_list.append(ar_recog)
            except Exception as e:
                pass


        if len(uf1_list) != 0:
            UF1 = np.mean(uf1_list)
        else:
            UF1 = 0
        if len(uar_list) != 0:
            UAR = np.mean(uar_list)
        else:
            UAR = 0
        return UF1, UAR
    except:
        return '', ''



class train_base():
    def __init__(self,epoch,train_loader,test_loader,net,param_path='./param_motion/16.pth'):
        self.epoch = epoch
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.param_path = param_path
        self.net = net.cuda()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)
        self.uar,self.acc,self.uf1=0,0,0

    def accuary(self,preds, labels):
        sum = 0
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                sum += 1
        return sum / len(preds)

    def score2category(self,preds):
        category = []
        for i in range(len(preds)):
            category.append(torch.argmax(preds[i]))
        return category

    def train_processing(self):


        pretrained_state = torch.load(self.param_path)
        model_dict = self.net.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            (k in model_dict and v.shape == model_dict[k].shape)}

        model_dict.update(pretrained_state)
        self.net.load_state_dict(model_dict)


        len_t = len(self.train_loader)
        len_v = len(self.test_loader)

        for i in range(self.epoch):
            v_loss, v_sum_acc, v_sum_uf1, v_sum_uar = 0, 0, 0, 0
            loss, t_loss, sum_acc, sum_uf1, sum_uar = 0, 0, 0, 0, 0
            j = 0

            for imgs, labels in self.train_loader:
                self.net.train()
                if self.param_path == './param_motion/resnet18.pth':
                    x = self.net(imgs[1].cuda())
                else:
                    x = self.net(imgs[0].cuda(), imgs[1].cuda())
                # print(x.size())
                weight = torch.tensor([0.33, 0.33, 0.34])
                loss = nn.functional.cross_entropy(x, labels.cuda(), weight=weight.cuda())

                preds = self.score2category(x)
                acc = self.accuary(preds, labels)

                uf1, uar = recognition_evaluation(labels, preds)
                # uf1 = metrics.f1_score(labels.cpu(), torch.tensor(np.array([i.cpu() for i in preds])), average='macro',
                #                        zero_division="warn")
                # uar = metrics.recall_score(labels.cpu(), torch.tensor(np.array([i.cpu() for i in preds])),
                #                            average='macro', zero_division="warn")
                t_loss += loss
                sum_acc += acc
                sum_uf1 += uf1
                sum_uar += uar

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                j += 1
                if (j + 1) % 10 == 0:
                    print(
                        ' j : {} loss:  {:.4F} acc:  {:.4F}  uar:  {:.4F}  uf1:  {:.4F}'
                            .format(int((j + 1) / 10), t_loss / j, sum_acc / j, sum_uar / j, sum_uf1 / j))

            with torch.no_grad():
                for imgs, labels in self.test_loader:
                    self.net.eval()
                    if self.param_path == './param_motion/resnet18.pth':
                        x = self.net(imgs[1].cuda())
                    else:
                        x = self.net(imgs[0].cuda(), imgs[1].cuda())
                    weight = torch.tensor([0.33, 0.33, 0.34])
                    loss = nn.functional.cross_entropy(x, labels.cuda(), weight=weight.cuda())

                    preds = self.score2category(x)
                    uf1, uar = recognition_evaluation(labels, preds)
                    acc = self.accuary(preds, labels)
                    # uf1 = metrics.f1_score(labels.cpu(), torch.tensor(np.array([i.cpu() for i in preds])),
                    #                        average='macro', zero_division="warn")
                    # uar = metrics.recall_score(labels.cpu(), torch.tensor(np.array([i.cpu() for i in preds])),
                    #                            average='macro',
                    #                            zero_division="warn")

                    v_loss += loss
                    v_sum_acc += acc
                    v_sum_uf1 += uf1
                    v_sum_uar += uar

            print(
                'epoch: {} loss: {:.4F} acc: {:.4F} uar: {:.4F}  uf1: {:.4F} || loss{:.4F} acc: {:.4F} uar: {:.4F}  uf1: {:.4F}'
                    .format(i, t_loss / len_t, sum_acc / len_t, sum_uar / len_t, sum_uf1 / len_t, v_loss / len_v,
                            v_sum_acc / len_v, v_sum_uar / len_v, v_sum_uf1 / len_v))

            if v_sum_uf1/len_v > self.uf1:
                self.uar = v_sum_uar/len_v
                self.uf1 = v_sum_uf1/len_v
                self.acc = v_sum_acc/len_v

            if self.uf1 == self.uf1 ==self.acc==1:
                return



