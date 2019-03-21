# reference: https://github.com/fxia22/kdnet.pytorch.git

import argparse
import time

import torch.utils.data

import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from model.dataset import *
# from model.pointnet import PointNetCls, PointNetClsDeeper
# from model.pointnet2_msg_cls import Pointnet2MSG
# from model.pointnet2_all import PointNet2ClsSsg, PointNet2ClsMsg
from model.train import KDNet
from model.kdtree import make_cKDTree
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


grasp_points_num = 1000
thresh_good = 0.6
thresh_bad = 0.6
point_channel = 3


def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2**31-1))


def my_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def train(model, loader, epoch, optimizer, scheduler):
    scheduler.step()
    model.train()
    torch.set_grad_enabled(True)
    correct = 0
    dataset_size = 0
    ij = 0
    for batch_idx, (data, target) in enumerate(loader):
        dataset_size += data.shape[0]
        # print(data.shape)
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        num_points = 2048
        #pre-processing for kdtree model :
        for i in range(data.shape[0]):
            point_set = data[i,:,:].transpose(1,0)
            t = target[i:i+1]
            point_set = point_set[:num_points]
            if point_set.size(0) <num_points :
                point_set = torch.cat([point_set, point_set[0:num_points - point_set.size(0)]], 0 )
            cutdim, tree =make_cKDTree(point_set.cpu().numpy(), depth = 11)
            cutdim_v = [(torch.from_numpy(np.array(item).astype(np.int64))) for item in cutdim]
            points = torch.FloatTensor(tree[-1])
            points_v = Variable(torch.unsqueeze(torch.squeeze(points), 0 )).transpose(2,1).cuda()
            output = model(points_v, cutdim_v)
            pred = output.data.max(1)[1] 
            loss = F.nll_loss(output, t)
            loss.backward()
            correct+=pred.eq(t.view_as(pred)).long().cpu().sum()
        optimizer.step()
        if (batch_idx+1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(
                epoch, batch_idx * args.batch_size, len(loader.dataset),
                100. * batch_idx * args.batch_size / len(loader.dataset), loss.item(), args.tag))
            logger.add_scalar('train_loss', loss.cpu().item(), batch_idx + epoch * len(loader))
            
        ij +=1
        if ij>60 :
            return float(correct)/float(dataset_size)
    
    return float(correct)/float(dataset_size)


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss = 0
    correct = 0
    dataset_size = 0
    da = {}
    db = {}
    res = []
    ij = 0
    for data, target, obj_name in loader:
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        num_points = 2048
        #pre-processing for kdtree model :
        for i in range(data.shape[0]):
            point_set = data[i,:,:].transpose(1,0)
            t = target[i:i+1]
            point_set = point_set[:num_points]
            if point_set.size(0) <num_points :
                point_set = torch.cat([point_set, point_set[0:num_points - point_set.size(0)]], 0 )
            cutdim, tree =make_cKDTree(point_set.cpu().numpy(), depth = 11)
            cutdim_v = [(torch.from_numpy(np.array(item).astype(np.int64))) for item in cutdim]
            points = torch.FloatTensor(tree[-1])
            points_v = Variable(torch.unsqueeze(torch.squeeze(points), 0 )).transpose(2,1).cuda()
            output = model(points_v, cutdim_v)
            pred = output.data.max(1)[1]    
            test_loss += F.nll_loss(output, t)            
            correct+=pred.eq(t.view_as(pred)).long().cpu().sum()
        for i, j, k in zip(obj_name, pred.data.cpu().numpy(), target.data.cpu().numpy()):
            res.append((i, j, k))
        
        ij +=1
        if ij>60 :
            test_loss /= len(loader.dataset)
            acc = float(correct)/float(dataset_size)
            return acc, test_loss

    test_loss /= len(loader.dataset)
    acc = float(correct)/float(dataset_size)
    return acc, test_loss


def main(train_loader, test_loader, optimizer, scheduler):
    if args.mode == 'train':
        for epoch in range(is_resume*args.load_epoch, args.epoch):
            acc_train = train(model, train_loader, epoch, optimizer, scheduler)
            print('Train done, acc={}'.format(acc_train))
            acc, loss = test(model, test_loader)
            print('Test done, acc={}, loss={}'.format(acc, loss))
            logger.add_scalar('train_acc', acc_train, epoch)
            logger.add_scalar('test_acc', acc, epoch)
            logger.add_scalar('test_loss', loss, epoch)
            if (epoch+1) % args.save_interval == 0:
                path = os.path.join(args.model_path, args.tag + '_{}.model'.format(epoch+1))
                torch.save(model, path)
                print('Save model @ {}'.format(path))
    else:
        print('testing...')
        acc, loss = test(model, test_loader)
        print('Test done, acc={}, loss={}'.format(acc, loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pointnetGPD')
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--load-model', type=str, default='')
    parser.add_argument('--load-epoch', type=int, default=-1)
    parser.add_argument('--model-path', type=str, default='./assets/learned_models',
                        help='pre-trained model path')
    parser.add_argument('--data-path', type=str, default='./data', help='data path')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=50)

    args = parser.parse_args()
    args.cuda = args.cuda if torch.cuda.is_available() else False

    if args.cuda:
        torch.cuda.manual_seed(1)

    logger = SummaryWriter(os.path.join('./assets', 'log', args.tag))  # WINDOWS
    np.random.seed(int(time.time()))

    train_loader = torch.utils.data.DataLoader(
        PointGraspOneViewDataset(
            grasp_points_num=grasp_points_num,
            path=args.data_path,
            tag='train',
            grasp_amount_per_file=480,
            thresh_good=thresh_good,
            thresh_bad=thresh_bad,
        ),
        batch_size=args.batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collate,
    )

    test_loader = torch.utils.data.DataLoader(
        PointGraspOneViewDataset(
            grasp_points_num=grasp_points_num,
            path=args.data_path,
            tag='test',
            grasp_amount_per_file=480,
            thresh_good=thresh_good,
            thresh_bad=thresh_bad,
            with_obj=True,
        ),
        batch_size=args.batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collate,
    )

    is_resume = 0
    if args.load_model and args.load_epoch != -1:
        is_resume = 1

    if is_resume or args.mode == 'test':
        model = torch.load(os.path.join(args.model_path, args.load_model), map_location='cuda:{}'.format(args.gpu))
        model.device_ids = [args.gpu]
        print('load model {}'.format(args.load_model))
    else:
        # model = PointNetClsDeeper(num_points=grasp_points_num, input_chann=point_channel, k=2)
        # model = Pointnet2MSG(num_classes=2)
        # model = PointNet2ClsSsg(num_classes=2)
        model = KDNet(k=2).cuda()
    if args.cuda:
        if args.gpu != -1:
            print("Running cuda on device {}".format(args.gpu))
            torch.cuda.set_device(args.gpu)
            model = model.cuda()
        else:
            device_id = [0, 1, 2, 3]
            torch.cuda.set_device(device_id[0])
            model = nn.DataParallel(model, device_ids=device_id).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    main(train_loader, test_loader, optimizer, scheduler)
