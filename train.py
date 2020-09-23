

import torch
import torch.functional as F
import torch.nn as nn

from util import TUTDataset
from model import SELDNet


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = SELDNet()
model.to(device)

tutdata = TUTDataset("data/mic_dev")
