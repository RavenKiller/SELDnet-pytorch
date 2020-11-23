import torch
import torch.nn as nn
import torch.nn.functional as F
from util import getDFTFeature

class SELDNet(nn.Module):
    def __init__(self,C=4,P=16,Q=16,R=16,K=10,feature_dim=512,cnn_layers=3,rnn_layers=1):
        super().__init__()
        self.C = C # the number of audio channels
        self.P = P # the number of feature maps
        self.Q = Q # the number of RNN hidden units
        self.R = R # the dimension of first full connection
        self.K = K # the number of classes
        self.feature_dim = feature_dim # the dimension of input features
        self.cnn_layers = cnn_layers # the number of CNN layers
        self.rnn_layers = rnn_layers # the number of RNN layers
        ##########################################
        # CNN part
        ##########################################
        self.convs = [nn.Conv2d(in_channels=2*C,out_channels=P,
                        kernel_size=(3,3),padding=(1,1))]
        for _ in range(cnn_layers-1):
            self.convs.append(nn.Conv2d(in_channels=P,out_channels=P,
                        kernel_size=(3,3),padding=(1,1)))

        # name all layers
        for i, conv in enumerate(self.convs):
            self.add_module("conv{}".format(i), conv)
            self.add_module("bn{}".format(i), nn.BatchNorm2d(P, affine=False))


        ##########################################
        # RNN part
        ##########################################
        self.grus = nn.GRU(input_size=2*P,hidden_size=Q,num_layers=rnn_layers,batch_first=True,bidirectional=True)
        self.add_module("grus",self.grus)


        ##########################################
        # SED part
        ##########################################
        self.sed_fc0 = nn.Linear(2*Q,R)
        self.sed_fc1 = nn.Linear(R,K)
        self.add_module("sed_fc0",self.sed_fc0)
        self.add_module("sed_fc1",self.sed_fc1)

        ##########################################
        # DOA part
        ##########################################
        self.doa_fc0 = nn.Linear(2*Q,R)
        self.doa_fc1 = nn.Linear(R,3*K)
        self.add_module("doa_fc0",self.doa_fc0)
        self.add_module("doa_fc1",self.doa_fc1)


        
        
    
    def forward(self,x):
        # x shape (N,2*C,T,M)
        for i in range(self.cnn_layers):
            y = F.relu(getattr(self,"conv{}".format(i))(x))
            if i<self.cnn_layers-1:
                y = F.max_pool2d(getattr(self,"bn{}".format(i))(y),kernel_size=(1,7))
            else:
                y = F.max_pool2d(getattr(self,"bn{}".format(i))(y),kernel_size=(1,5))
            x = y
        # final y shape (N,P,T,2)

        y = y.permute(0,2,3,1) # shape(N,T,2,P)
        y = y.contiguous().view(y.shape[0],y.shape[1],-1) # shape (N,T,2*P)
        # h0 = torch.rand((2*self.rnn_layers,y.shape[0],self.Q)).cuda()
        y, _ = self.grus(y) # shape(N,T,2*Q)
        y = torch.tanh(y) # shape(N,T,2*Q)

        sed_y = torch.sigmoid(self.sed_fc1(self.sed_fc0(y)))
        doa_y = torch.tanh(self.doa_fc1(self.doa_fc0(y)))
        return sed_y, doa_y
if __name__ == "__main__":
    tmp = SELDNet()

    FILENAME = "data/mic_dev/split1_ir1_ov1_24.wav"
    feature = getDFTFeature(FILENAME)
    print(feature.shape)
    feature = feature.unsqueeze(0)
    output = tmp(feature)
    print(output[0].shape)
    print(output[1].shape)
    print(output[1])

