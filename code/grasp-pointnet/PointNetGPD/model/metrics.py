#from the PyTorch implementation of DGCNN from https://github.com/ashawkey/dgcnn.pointCloud.pytorch

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class config:
    def __init__(self):
        self.cuda = True
        self.epochs = 1
        self.learning_rate = 0.01
        self.batch_size = 16
        self.regularization = 0
        self.nCls = 10
        self.Fin = 3
        self.dropout = 0.5
        self.negative_slope = 0.2
    def __str__(self):
        s = "Configurations:\n"
        for attr, val in vars(self).items():
            s += "{:>20}: {}\n".format(attr, val)
        return s

def get_loss_cls(preds, labels, outs):
    ce = F.cross_entropy(preds, labels)
    if outs is None:
        return ce
    mat = outs['transform']
    B, d, d = mat.shape
    mat = torch.bmm(mat, mat.permute(0, 2, 1))
    eye = torch.eye(d).unsqueeze(0).repeat(B, 1, 1).to(mat.device)
    l2 = F.mse_loss(mat, eye)
    return ce + 1e-3*l2

def accuracy(logits, labels, verbose=False):
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(labels, preds)
    res = preds==labels
    ncorrect = torch.nonzero(res).shape[0]
    ntotal = res.shape[0]
    if verbose:
        print("preds:", preds)
        print("labels", labels)
    return acc, ncorrect, ntotal

if __name__ == "__main__":
    conf = config()
    print(conf)
