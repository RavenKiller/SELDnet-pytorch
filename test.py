

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from util import TUTDataset
from model import SELDNet

import argparse
import sys
import os
from pprint import pprint

IR_SET = ["ir0","ir1","ir2","ir3","ir4"]
SPLIT_SET = ["split4"]
OV_SET = ["ov1"]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def test(args):
    '''The function to train SELDNet
    Args:
        args: cmd line arguments parsed by `argparse`
            batch_size
            epoch_num
    '''
    tutdata = TUTDataset("data/mic_dev","data/metadata_dev",sample_freq=44100,split_set=SPLIT_SET,ir_set=IR_SET,ov_set=OV_SET)
    tutloader = data.DataLoader(tutdata,batch_size=args.batch_size,shuffle=True)
    criterion_sed = nn.CrossEntropyLoss()
    criterion_doa = nn.MSELoss()


    model = SELDNet(K=tutdata.num_class)
    model.load_state_dict(torch.load("SELDNet-best.ckpt"))
    model.to(device)
    model.eval()

    test_loss_sum = 0
    steps = 0
    for sample,sed,doa in tutloader:
        sample = sample.to(device)
        sed = sed.to(device)
        doa = doa.to(device)
        print("steps {}".format(steps))
        out_sed,out_doa = model(sample)
        out_sed = out_sed.reshape(-1,tutdata.num_class)
        sed = sed.reshape(-1)
        loss_sed = criterion_sed(out_sed,sed)
        loss_doa = criterion_doa(out_doa.double(),doa.double())
        loss = loss_sed+loss_doa
        test_loss_sum+=float(loss)
        steps+=1
    print("test loss is {}".format(test_loss_sum/steps))
def test_decode():
    tutdata = TUTDataset("data/mic_dev","data/metadata_dev",sample_freq=44100,split_set=SPLIT_SET,ir_set=IR_SET,ov_set=OV_SET)
    print(tutdata.file_names[0])
    sample,sed,doa = tutdata[0]
    sed_onehot = torch.zeros((sed.shape[0],tutdata.num_class))
    print(sed)
    print(set(list(sed.numpy())))
    print(tutdata.name2idx)
    for k,v in enumerate(sed):
        sed_onehot[k,v] = 1
    res = tutdata.decode_one(sed_onehot,doa)
    pprint(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SELDNet params')
    parser.add_argument('--batch_size', type=int,default=4,help='The batch size')
    args = parser.parse_args()
    test_decode()


