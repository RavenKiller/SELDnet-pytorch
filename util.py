import torchaudio
import torch
import torch.utils.data as data

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
def getDFTFeature(filepath,win_size=1024,win_shift=512,preemphasis=False,channel_first=True,drop_dc=True,cut_len=5000):
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
    log_power_spectrum = np.log10(power_spectrum)*10
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
    def __init__(self, data_folder):
        '''
        Args:
            data_folder: the path where audio files are stored in
        '''
        super(TUTDataset).__init__()
        self.data_folder = data_folder
        self.file_names = list(os.listdir(data_folder))

    def __getitem__(self,index):
        file_name = self.file_names[index]
        return getDFTFeature(os.path.join(self.data_folder,file_name))

    def __len__(self):
        return len(self.file_names)

if __name__ == "__main__":
    tutdata = TUTDataset("data/mic_dev")
    loader = data.DataLoader(tutdata,batch_size=8)
    for v in loader:
        print(v.shape)