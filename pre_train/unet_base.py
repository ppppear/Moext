import torch
import torch.nn as nn



class conv_block(nn.Module):
    def __init__(self,in_channel,out_channel,max_pooling = True,up_sample = False,times = 1):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel*times,out_channel*times,3,1,1)
        self.conv2 = nn.Conv2d(out_channel*times,out_channel,3,1,1)
        self.bn1 = nn.BatchNorm2d(out_channel*times)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        if max_pooling:
            self.pooling = nn.MaxPool2d(2,2) if max_pooling else nn.AvgPool2d(2,2)
        if up_sample:
            self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self,x):
        # print("--------------")
        x = self.relu(self.conv1(x))
        x = self.bn1(x)

        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        if hasattr(self,'pooling'):
            x = self.pooling(x)
        elif hasattr(self,'up'):
            x = self.up(x)
        # print(x.shape)
        return x


class feature_separation(nn.Module):
    def __init__(self):
        super(feature_separation, self).__init__()
        self.up_sample_motion = nn.ModuleList()
        param_list = [512,512,256,128,32]
        for i in range(4):
            self.up_sample_motion.append(conv_block(param_list[i],param_list[i+1],max_pooling = False,  up_sample=True if i > 0 else False,times = 2 if i > 1 else 1))
        self.up_sample_texture = nn.ModuleList()
        for i in range(4):
            self.up_sample_texture.append(conv_block(param_list[i],param_list[i+1],max_pooling=False,up_sample=True if i > 0 else False,times = 2 if i > 1 else 1))


    def forward(self,res):
        res_texture = res_motion = res[-1]
        # print(res[0].shape,res[1].shape)
        for i in range(4):
            # print("seqaration i: ",i)
            # print(res_texture.shape)
            res_texture = self.up_sample_texture[i](res_texture)
            res_motion = self.up_sample_motion[i](res_motion)

            if i > 0 and i < 3:
                res_texture = torch.cat([res_texture,res[2 - i]],dim = 1)
                res_motion = torch.cat([res_motion, res[2 - i]], dim = 1)


        return res_motion,res_texture


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = nn.ModuleList()
        param_list = [3,64,128,256,512,512]
        for i in range(4):
            self.backbone.append(conv_block(param_list[i],param_list[i+1])) # 14*14*512
        self.feature_separation = feature_separation()

    def forward(self,x):
        res_base = []
        for i in range(4):

            x = self.backbone[i](x)
            if i > 0:
                res_base.append(x)
            # print("i: ",x.size())
        motion,texture = self.feature_separation(res_base)
        return motion,texture


class Manipulator(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(Manipulator, self).__init__()
        self.conv1 = conv_block(inchannel,inchannel,max_pooling=False)
        self.conv2 = conv_block(inchannel,outchannel,max_pooling=False)
        self.relu = nn.ReLU()

    def forward(self,onset,apex):

        motion = apex - onset
        t = motion
        # print("Manipulator input : ",motion.size())
        motion = self.conv2(self.conv1(motion))
        motion += t
        # motion = motion + apex
        # print("Manipulator output: ",motion.size())
        return motion





class Decoder(nn.Module):
    def __init__(self, dim_in=32, dim_out=3, num_resblk=8):
        super(Decoder, self).__init__()


        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(64,32,(3,3),(1,1),1,bias=False),
            nn.ReLU(),
            nn.Conv2d(32,32,(3,3),(1,1),1,bias=False),
            nn.ReLU(),)

        self.res_layer = nn.ModuleList()
        for i in range(4):
            self.res_layer.append(conv_block(32,32,False))
        self.res_layer = nn.Sequential(*self.res_layer)
        self.conv_layer_2 = nn.Conv2d(32,3,(7,7),(1,1),3,bias=False)
    def forward(self, texture, motion):

        x = torch.cat([texture, motion], 1)
        x = self.conv_layer_1(x)
        x = self.res_layer(x)
        x = self.up(x)
        x = self.conv_layer_2(x)


        return x

class Motion_net(nn.Module):
    def __init__(self):
        super(Motion_net, self).__init__()
        self.encoder = Encoder()
        self.manipulator = Manipulator(32,32)
        self.decoder = Decoder()

    def forward(self,onset,apex,rotate):
        motion_onset,texture_onset = self.encoder(onset)
        motion_apex,texture_apex = self.encoder(apex)
        motion_rotate,texture_rotate = self.encoder(rotate)
        motion = self.manipulator(motion_onset,motion_apex)
        new_apex = self.decoder(texture_onset,motion)

        # t_ar = [texture_apex-texture_rotate]
        m_t = [motion_apex , texture_apex]
        t_oa = [texture_onset,texture_apex]
        m_ar = [motion_apex,motion_rotate]
        m_ao = [motion_onset, motion_apex]


        return new_apex,m_t,t_oa,m_ar,m_ao


# model = Motion_net()
# Onset = torch.ones(1, 3, 224, 224)
# Apex = torch.ones(1, 3, 224, 224)
#
# model(Onset,Apex,Apex)
