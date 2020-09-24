

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from util import TUTDataset
from model import SELDNet

IR_SET = ["ir0","ir1","ir2","ir3","ir4"]
SPLIT_SET = ["split1","split2"]
OV_SET = ["ov1"]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tutdata = TUTDataset("data/mic_dev","data/metadata_dev",sample_freq=44100,split_set=SPLIT_SET,ir_set=IR_SET,ov_set=OV_SET)
tutloader = data.DataLoader(tutdata,batch_size=8,shuffle=True)
print(torch.cuda.is_available())
model = SELDNet(K=tutdata.num_class)
model = model.to(device)
model.train()
print(model)
criterion_sed = nn.CrossEntropyLoss()
criterion_doa = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

epoch_num = 10

for epoch in range(epoch_num):
    for sample,sed,doa in tutloader:
        sample = sample.to(device)
        sed = sed.to(device)
        doa = doa.to(device)
        optimizer.zero_grad()
        out_sed,out_doa = model(sample)
        out_sed = out_sed.reshape(-1,tutdata.num_class)
        sed = sed.reshape(-1)
        loss_sed = criterion_sed(out_sed,sed)
        loss_doa = criterion_doa(out_doa.double(),doa.double())
        loss = loss_sed+loss_doa
        print(loss)
        loss.backward()
        optimizer.step()
        break

