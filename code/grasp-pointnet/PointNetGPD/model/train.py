import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

num_points = 2048


class KDNet(nn.Module):
    def __init__(self, k=16):
        super(KDNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 8 * 3, 1, 1)
        self.conv2 = nn.Conv1d(8, 32 * 3, 1, 1)
        self.conv3 = nn.Conv1d(32, 64 * 3, 1, 1)
        self.conv4 = nn.Conv1d(64, 64 * 3, 1, 1)
        self.conv5 = nn.Conv1d(64, 64 * 3, 1, 1)
        self.conv6 = nn.Conv1d(64, 128 * 3, 1, 1)
        self.conv7 = nn.Conv1d(128, 256 * 3, 1, 1)
        self.conv8 = nn.Conv1d(256, 512 * 3, 1, 1)
        self.conv9 = nn.Conv1d(512, 512 * 3, 1, 1)
        self.conv10 = nn.Conv1d(512, 512 * 3, 1, 1)
        self.conv11 = nn.Conv1d(512, 1024 * 3, 1, 1)
        self.fc = nn.Linear(1024, k)

    def forward(self, x, c):
        def kdconv(x, dim, featdim, sel, conv):
            batchsize = x.size(0)
            #print(batchsize)
            x = F.relu(conv(x))
            x = x.view(-1, featdim, 3, dim)
            x = x.view(-1, featdim, 3 * dim)
            sel = Variable(sel + (torch.arange(0, dim) * 3).long())            
            if x.is_cuda:
                sel = sel.cuda()
            x = torch.index_select(x, dim=2, index=sel)
            x = x.view(-1, featdim, int((dim / 2)), 2)
            x = torch.squeeze(torch.max(x, dim=-1, keepdim=True)[0], 3)
            return x

        x1 = kdconv(x, 2048, 8, c[0], self.conv1)
        x2 = kdconv(x1, 1024, 32,c[1], self.conv2)
        x3 = kdconv(x2, 512, 64, c[2], self.conv3)
        x4 = kdconv(x3, 256, 64, c[3], self.conv4)
        x5 = kdconv(x4, 128, 64, c[4], self.conv5)
        x6 = kdconv(x5, 64, 128, c[5], self.conv6)
        x7 = kdconv(x6, 32, 256, c[6], self.conv7)
        x8 = kdconv(x7, 16, 512, c[7], self.conv8)
        x9 = kdconv(x8, 8, 512, c[8], self.conv9)
        x10 = kdconv(x9, 4, 512, c[9], self.conv10)
        x11 = kdconv(x10, 2, 1024, c[10], self.conv11)
        x11 = x11.view(-1, 1024)
        out = F.log_softmax(self.fc(x11),dim=1)
        return out

