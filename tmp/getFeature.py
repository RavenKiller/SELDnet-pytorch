import torchaudio
import torch
FILENAME = "split1_ir0_ov1_1.wav"
waveform, sample_freq = torchaudio.load(FILENAME)
print(sample_freq)
print(waveform.shape)

win_size = 1024
win_shift = 512
m, n = waveform.shape

# padding to 2^k
if (n-win_size)%win_shift != 0:
    waveform = torch.cat([waveform,torch.zeros(m,win_shift-(n-win_size)%win_shift)],dim=1)
    n = waveform.shape[1]

# split frames into rows
frame_num = (n-win_size)//win_shift + 1
strided_input = waveform.as_strided((frame_num,win_size),(win_shift,1))
print(strided_input.shape)
strided_input = strided_input - torch.mean(strided_input,dim=1).unsqueeze(1)


# 
fft = torch.rfft(waveform,1)
print(fft.shape)
