import torchaudio
import torch
import torch.utils.data as data

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import math
def getDFTFeature(filepath,win_size=1024,win_shift=512,preemphasis=False,channel_first=True,drop_dc=True,cut_len=5160,normalize=False):
    '''
    获取一个音频的对数DFT频谱
    
    Args:
        filepath: 音频路径
        win_size: 窗口大小（点）
        win_shift: 滑动距离（点）
        preemphasis: 是否预强化。通过一阶差分弱低频强高频
        channel_first: whether to put channels in first dimension
        drop_dc: whether to drop DC component in spectrum (frequency==0)
        cut_len: keep the fix number of points in time axis
        normalize: 观察发现能量很小，对数能量在-100的数量级，有必要处理一下
        
    Return:
        (log_power_spectrum, phase_spectrum):能量谱与相位谱相叠形成的tensor
        大小原本为(2C,T,M//2)，C为通道数据,T为帧数,M为FFT点数
        经转置后变为(T,M//2,2C)
    '''
    waveform, sample_freq = torchaudio.load(filepath)
    m, n = waveform.shape
    # padding to 2^k
    if (n-win_size)%win_shift != 0:
        waveform = torch.cat([waveform,torch.zeros(m,win_shift-(n-win_size)%win_shift)],dim=1)
        n = waveform.shape[1]
    
    # split frames into rows
    frame_num = (n-win_size)//win_shift + 1
    strided_input = waveform.as_strided((m,frame_num,win_size),(n,win_shift,1))
    strided_input = strided_input - torch.mean(strided_input,dim=2).unsqueeze(2)
    
    # pre-emphasis
    preemphasis = 0.97
    offset_strided_input = torch.nn.functional.pad(
            strided_input, (1, 0), mode='replicate')
    strided_input = strided_input - preemphasis*offset_strided_input[:,:,:-1]
    
    # windowed and FFT
    win_func = torch.hamming_window(win_size,periodic=False)
    windowed_input = strided_input * win_func
    fft = torch.rfft(windowed_input,1,normalized=False, onesided=True)*2/win_size
    if drop_dc:
        fft = fft[:,:,1:]
    fft = fft[:,:cut_len,:]
    power_spectrum = fft.pow(2).sum(3)
    log_power_spectrum = torch.log10(power_spectrum)*10
    # 对于能量谱正则化处理
    mean_vec = log_power_spectrum.mean(axis=1,keepdim=True)
    std_vec = log_power_spectrum.std(axis=1,keepdim=True)
    log_power_spectrum = (log_power_spectrum-mean_vec)/std_vec
    # 
    phase_spectrum = fft[:,:,:,0]/fft.pow(2).sum(3).sqrt()
    
    phase_spectrum = torch.acos(phase_spectrum)
    phase_spectrum[fft[:,:,:,0]<0] = -phase_spectrum[fft[:,:,:,0]<0]
    spectrums = torch.cat([log_power_spectrum,phase_spectrum],dim=0)
    if not channel_first:
        spectrums = spectrums.permute(1,2,0)
    return spectrums



class TUTDataset(data.Dataset):
    '''
    TUT audio dataset
    '''
    def __init__(self, data_folder,label_folder,sample_freq=44100,split_set=["split1"],ir_set=["ir0"],ov_set=["ov1"],cut_len=5160):
        '''
        Args:
            data_folder: the path where audio files stored in
            label_folder: the path where label files stored in
            sample_freq: the sample frequency of audios
            split_set: a list, containing the split sets to be loaded
            ir_set: a list, containing the ir sets to be loaded
            ov_set: a list, containing the ov sets to be loaded
            cut_len: the length of feature in time axis
        '''
        super(TUTDataset).__init__()
        self.data_folder = data_folder
        self.label_folder = label_folder
        self.file_names = list(os.listdir(data_folder))
        self.sample_freq = sample_freq
        self.cut_len = cut_len
        
        # choose target set
        self.split_set = set(split_set) 
        self.ir_set = set(ir_set)
        self.ov_set = set(ov_set)
        self.file_names = [name for name in self.file_names 
                            if (set(name.split("_"))&self.split_set 
                            and set(name.split("_"))&self.ir_set
                            and set(name.split("_"))&self.ov_set)]


        self.name2idx, self.idx2name = self.getAllEvents()
        self.name2idx['silence'] = 0
        self.idx2name.insert(0,'silence')
        self.num_class = len(self.idx2name)

        self.frame_len = 1024
        self.frame_shift = 512

    def __getitem__(self,index):
        '''get a data sample
        Args: 
            index: the index
        Return:
            (sample_data,sample_sed_label,sample_doa_label):
                sample_data: 2C x N x F matrix, C is the number of channels, N is time points, F is frequency bins.
                sample_sed_label: N-dim vector, the class of sound event
                sample_doa_label: N x 3K matrix,  the position of sound event (at unit circle), the final dimension is like (x0,y0,z0,x1,y1,z1,...)
        '''
        file_name = self.file_names[index]
        label_name = file_name.replace('.wav','.csv')
        feature = getDFTFeature(os.path.join(self.data_folder,file_name))
        label = self.getLabel(os.path.join(self.label_folder,label_name))
        if(feature.shape[1]<5160):
            feature = torch.cat((feature,torch.zeros(feature.shape[0],5160-feature.shape[1],feature.shape[2])),dim=1)
        return feature,torch.LongTensor(label[0]),torch.tensor(label[1])

    def getAllEvents(self):
        '''get all event labels
        Return:
            (label2idx,idx2label)
                label2idx: a dict to convert event name to index. events count froim 1
                idx2label: a list to convert index to event name. 0 is corresponding to silence
        '''
        event_set = set([])
        for filename in os.listdir(self.label_folder):
            if '.csv' in filename:
                with open(os.path.join(self.label_folder,filename),'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        event_set.add(row['sound_event_recording'])
        return {v:(k+1) for k,v in enumerate(event_set)},list(event_set)

    def decode_one(self,sed_tensor,doa_tensor):
        '''decode information of predict output
        Args:
            sed_tensor: a tensor of predict SED, size T x num_class+1
            doa_tensor: a tensor of predict DOA, size T x (num_class+1)*3
        Return:
            all_events: a list of all events, every element is a dict storing name, start, end, ele, azi, etc. of a event.
        '''
        sed_tensor = sed_tensor.cpu().numpy()
        doa_tensor = doa_tensor.cpu().numpy()
        all_events = []
        current_idx = set([])
        onset = {}
        N = sed_tensor.shape[0]
        K = sed_tensor.shape[1]
        for i in range(N):
            # print(sed_tensor[i,:])
            idx = set(np.where(sed_tensor[i,:]>0.5)[0])
            # open new events
            for v in idx:
                if v not in onset:
                    onset[v] = i
            # close old events
            tmp = onset.copy()
            for v in tmp.keys():
                if v not in idx:
                    startp = onset.pop(v)
                    endp = i
                    start = ((startp-1)*self.frame_shift+self.frame_len)/self.sample_freq
                    end = ((endp-1)*self.frame_shift+self.frame_len)/self.sample_freq
                    x = np.mean(doa_tensor[startp:endp,v*3])
                    y = np.mean(doa_tensor[startp:endp,v*3+1])
                    z = np.mean(doa_tensor[startp:endp,v*3+2])
                    r = math.sqrt(x*x+y*y+z*z)
                    if r<1e-5:
                        ele=0
                        azi=0
                    else:
                        ele = math.asin(z/r)
                        azi = math.acos(max(min(x/(r*math.cos(ele)),1),-1))
                        if y<0:
                            azi = -azi
                    all_events.append({
                        "idx":v,
                        "event":self.idx2name[v],
                        "start":start,
                        "end":end,
                        "x":x,
                        "y":y,
                        "z":z,
                        "ele":ele*180/math.pi,
                        "azi":azi*180/math.pi
                    })
        # events continuing to audio end
        i = N-1
        tmp = onset.copy()
        print(onset)
        for v in tmp.keys():
            startp = onset.pop(v)
            endp = i
            start = ((startp-1)*self.frame_shift+self.frame_len)/self.sample_freq
            end = ((endp-1)*self.frame_shift+self.frame_len)/self.sample_freq
            x = np.mean(doa_tensor[startp:endp,v*3])
            y = np.mean(doa_tensor[startp:endp,v*3+1])
            z = np.mean(doa_tensor[startp:endp,v*3+2])
            r = math.sqrt(x*x+y*y+z*z)
            if r<1e-5:
                ele=0
                azi=0
            else:
                ele = math.asin(z/r)
                azi = math.acos(max(min(x/(r*math.cos(ele)),1),-1))
                if y<0:
                    azi = -azi
            all_events.append({
                "idx":v,
                "event":self.idx2name[v],
                "start":start,
                "end":end,
                "x":x,
                "y":y,
                "z":z,
                "ele":ele*180/math.pi,
                "azi":azi*180/math.pi
            })
        
        return all_events
    
    def getLabel(self,filepath):
        '''get label information from a csv file
        Args:
            filepath: the relative file path
        Return:
            (sed, doa)
                sed: N x (num_class+1) matrix. every row is a one hot vector
                doa: N x (num_class+1)*3 matrix. every row is positions of all possible events, like (x0,y0,z0,x1,y1,z1,...)

        '''
        sed = np.zeros(self.cut_len)
        doa = np.zeros((self.cut_len,self.num_class*3))
        with open(filepath,'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['sound_event_recording']
                idx = self.name2idx[name]
                start = float(row['start_time'])
                end = float(row['end_time'])
                dist = float(row['dist'])
                sp = int(start*self.sample_freq)
                if sp<1024:
                    sp = 0
                else:
                    sp = (sp-1024)//512+1
                ep = int(end*self.sample_freq)
                if ep<1024:
                    ep = 0
                else:
                    ep = (ep-1024)//512+1
                ele = float(row['ele'])*np.pi/180
                azi = float(row['azi'])*np.pi/180

                sed[sp:ep] = idx
                doa[sp:ep,idx*3:(idx*3+3)] = np.array([[np.cos(ele)*np.cos(azi),np.cos(ele)*np.sin(azi),np.sin(ele)]])*dist

        return (sed,doa)
    def __len__(self):
        return len(self.file_names)

if __name__ == "__main__":
    tutdata = TUTDataset("data/mic_dev","data/metadata_dev",sample_freq=44100)
    loader = data.DataLoader(tutdata,batch_size=8,shuffle=False)
    for v,k,d in loader:
        print(v.shape)
        print(d)