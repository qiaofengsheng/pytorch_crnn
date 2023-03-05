from torch import nn
import torch

class BidirectionalLSTM(nn.Module):
 
    def __init__(self, nInput_size, nHidden,nOut):
        super(BidirectionalLSTM, self).__init__()
 
        self.lstm = nn.LSTM(nInput_size, nHidden, bidirectional=True)
        self.linear = nn.Linear(nHidden * 2, nOut)
 
    def forward(self, input):
        recurrent, (hidden,cell)= self.lstm(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
 
        output = self.linear(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1) #输出变换为[seq,batch,类别总数]
 
        return output
 
 
 
class CNN(nn.Module):
 
    def __init__(self,imageHeight,nChannel):
        super(CNN,self).__init__()
        assert imageHeight % 32 == 0,'image Height has to be a multiple of 32'
 
        self.depth_conv0 = nn.Conv2d(in_channels=nChannel,out_channels=nChannel,kernel_size=3,stride=1,padding=1,groups=nChannel)
        self.point_conv0 = nn.Conv2d(in_channels=nChannel,out_channels=64,kernel_size=1,stride=1,padding=0,groups=1)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=2,stride=2)
 
        self.depth_conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,groups=64)
        self.point_conv1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1,stride=1,padding=0,groups=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
 
        self.depth_conv2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,groups=128)
        self.point_conv2 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1,stride=1,padding=0,groups=1)
        self.batchNorm2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
 
        self.depth_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256)
        self.point_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,1),padding=(0,1))
 
        self.depth_conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256)
        self.point_conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1)
        self.batchNorm4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
 
        self.depth_conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.point_conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,1),padding=(0,1))
 
        #self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0)
        self.depth_conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0, groups=512)
        self.point_conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1)
        self.batchNorm6 = nn.BatchNorm2d(512)
        self.relu6= nn.ReLU(inplace=True)
 
    def forward(self,input):
        depth0 = self.depth_conv0(input)
        point0 = self.point_conv0(depth0)
        relu0 = self.relu0(point0)
        pool0 = self.pool0(relu0)
       # print(pool0.size())
 
        depth1 = self.depth_conv1(pool0)
        point1 = self.point_conv1(depth1)
        relu1 = self.relu1(point1)
        pool1 = self.pool1(relu1)
        #print(pool1.size())
 
        depth2 = self.depth_conv2(pool1)
        point2 = self.point_conv2(depth2)
        batchNormal2 = self.batchNorm2(point2)
        relu2 = self.relu2(batchNormal2)
        #print(relu2.size())
 
        depth3 = self.depth_conv3(relu2)
        point3 = self.point_conv3(depth3)
        relu3 = self.relu3(point3)
        pool3 = self.pool3(relu3)
        #print(pool3.size())
 
        depth4 = self.depth_conv4(pool3)
        point4 = self.point_conv4(depth4)
        batchNormal4 = self.batchNorm4(point4)
        relu4 = self.relu4(batchNormal4)
        #print(relu4.size())
 
        depth5 = self.depth_conv5(relu4)
        point5 = self.point_conv5(depth5)
        relu5 = self.relu5(point5)
        pool5 = self.pool5(relu5)
        #print(pool5.size())
 
        depth6 = self.depth_conv6(pool5)
        point6 = self.point_conv6(depth6)
        batchNormal6 = self.batchNorm6(point6)
        relu6 = self.relu6(batchNormal6)
        #print(relu6.size())
 
        return relu6
 
class CRNN(nn.Module):
    def __init__(self,imgHeight, nChannel, nClass, nHidden):
        super(CRNN,self).__init__()
 
        self.cnn = nn.Sequential(CNN(imgHeight, nChannel))
        self.lstm = nn.Sequential(
            BidirectionalLSTM(512, nHidden, nHidden),
            BidirectionalLSTM(nHidden, nHidden, nClass),
        )
    def forward(self,input):
        conv = self.cnn(input) # [B,C,1,W]
        # pytorch框架输出结构为BCHW
        batch,channel,height,width = conv.size()
        assert  height==1,"the output height must be 1."
        # 将height==1的维度去掉-->BCW
        conv = conv.squeeze(dim=2)
        # 调整各个维度的位置(B,C,W)->(W,B,C)，对应lstm的输入(seq,batch,input_size)
        conv = conv.permute(2,0,1)
 
        output = self.lstm(conv) # [W,B,num_classes]
        output = torch.log_softmax(output,dim=2)
        return output

