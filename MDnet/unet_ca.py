
from pre_train.unet_base import *

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class MultiScale_Conv(nn.Module):
    def __init__(self, channel):
        super(MultiScale_Conv, self).__init__()
        self.globel_conv_test = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=4*channel, padding=1, kernel_size=(3, 3)),
            nn.ReLU(),
        )
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.globel_conv_test(x)

        out = self.maxpooling(out)

        return out  # 4C * h/2 h/2

class Category(nn.Module):
    def __init__(self):
        super(Category, self).__init__()
        self.MultiScale_layer1 = MultiScale_Conv(32)
        self.MultiScale_layer2 = MultiScale_Conv(128)

        self.Conv_layer1 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2)

        )
        self.Conv_layer2=nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.Avg_fc = nn.Sequential(nn.AvgPool2d(7, 7),
                                    nn.Flatten(),
                                    nn.Linear(512, 128),
                                    )

        self.Fc2 =nn.Linear(128,3)


    def forward(self, x):
        x = self.MultiScale_layer1(x)
        x = self.MultiScale_layer2(x)
        x = self.Conv_layer1(x)
        # x = self.Conv_layer3(x)
        # x = self.Conv_layer4(x)
        x = self.Conv_layer2(x)
        print(000,x.size())
        x =self.Avg_fc(x)
        feature_spot = x
        x = self.Fc2(x)
        return x,feature_spot

class net(Motion_net):
    def __init__(self):
        super(net, self).__init__()
        self.category = Category()
        self.ca = ChannelAttention(32)
    def forward(self,onset,apex):
        motion_onset,texture_onset = self.encoder(onset)
        motion_apex,texture_apex = self.encoder(apex)
        motion = self.manipulator(motion_onset,motion_apex)
        motion = self.ca(motion) * motion
        print(motion.size())
        x,self.feature_spot = self.category(motion)

        return x
model = net()
Onset = torch.ones(1, 3, 224, 224)
Apex = torch.ones(1, 3, 224, 224)

print(model(Onset,Apex).size())

