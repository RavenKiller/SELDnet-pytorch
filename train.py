

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

IR_SET = ["ir0","ir1","ir2","ir3","ir4"]
SPLIT_SET = ["split1","split2"]
OV_SET = ["ov1"]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(args):
    '''The function to train SELDNet
    Args:
        args: cmd line arguments parsed by `argparse`
            batch_size
            epoch_num
    '''
    tutdata = TUTDataset("data/mic_dev","data/metadata_dev",sample_freq=44100,split_set=SPLIT_SET,ir_set=IR_SET,ov_set=OV_SET)
    tutloader = data.DataLoader(tutdata,batch_size=args.batch_size,shuffle=True)
    evaldata = TUTDataset("data/mic_dev","data/metadata_dev",sample_freq=44100,split_set=["split3"],ir_set=IR_SET,ov_set=OV_SET)
    evalloader = iter(data.DataLoader(evaldata,batch_size=args.batch_size,shuffle=True))
    sample_eval, sed_eval, doa_eval = next(evalloader)
    sample_eval = sample_eval.to(device)
    sed_eval = sed_eval.to(device)
    doa_eval = doa_eval.to(device)


    model = SELDNet(K=tutdata.num_class)
    model.to(device)
    model.train()
    print(model)
    criterion_sed = nn.CrossEntropyLoss()
    criterion_doa = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    epoch_num = args.epoch_num
    min_loss = 100
    train_loss_sum = 0
    steps = 0
    save_step = 5
    for epoch in range(epoch_num):
        for sample,sed,doa in tutloader:
            sample = sample.to(device)
            sed = sed.to(device)
            doa = doa.to(device)
            print("steps {}".format(steps))
            optimizer.zero_grad()
            out_sed,out_doa = model(sample)
            out_sed = out_sed.reshape(-1,tutdata.num_class)
            sed = sed.reshape(-1)
            loss_sed = criterion_sed(out_sed,sed)
            loss_doa = criterion_doa(out_doa.double(),doa.double())
            loss = loss_sed+loss_doa
            train_loss_sum += float(loss)
            loss.backward()
            optimizer.step()

            # evaluation
            if steps % save_step==0:
                model.eval()
                with torch.no_grad():
                    out_sed,out_doa = model(sample_eval)
                    out_sed = out_sed.reshape(-1,tutdata.num_class)
                    sed_eval = sed_eval.reshape(-1)
                    loss_sed = criterion_sed(out_sed,sed_eval)
                    loss_doa = criterion_doa(out_doa.double(),doa_eval.double())
                    loss = loss_sed+loss_doa
                    print("train loss: {}, evaluation loss: {}".format(float(train_loss_sum)/save_step,float(loss)))
                    if float(loss)<min_loss:
                        print("saveing model...")
                        min_loss = float(loss)
                        torch.save(model.state_dict(),"SELDNet-best.ckpt")
                    train_loss_sum = 0
                model.train()

            steps+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SELDNet params')
    parser.add_argument('--epoch_num', type=int,default=4,help='The number of total training epochs')
    parser.add_argument('--batch_size', type=int,default=4,help='The batch size')
    args = parser.parse_args()
    train(args)


