#modified from the PyTorch implementation of PointNetGPD (https://github.com/lianghongzhuo/PointNetGPD)

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
from model.pointnet2_all import PointNet2ClsSsg, PointNet2ClsMsg
from model.dgcnn import *
from model.metrics import *
import numpy
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
    for batch_idx, (data, target) in enumerate(loader):
        dataset_size += data.shape[0]
        # print(data.shape)
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        #print('output shape is {} and is {}'.format(output.shape,output))
        #print('target shape is {} and is {}'.format(target.shape,target))
        #output_np = output.detach().numpy()
        #print('output np array is {}'.format(output_np))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        #print("pred is {}".format(pred))
        #loss = F.nll_loss(pred,target)
        #loss.backward()
        #optimizer.step()
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        if (batch_idx+1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(
                epoch, batch_idx * args.batch_size, len(loader.dataset),
                100. * batch_idx * args.batch_size / len(loader.dataset), loss.item(), args.tag))
            logger.add_scalar('train_loss', loss.cpu().item(), batch_idx + epoch * len(loader))
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
    for data, target, obj_name in loader:
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)  # N*C
        test_loss += F.nll_loss(output, target, size_average=False).cpu().item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        for i, j, k in zip(obj_name, pred.data.cpu().numpy(), target.data.cpu().numpy()):
            res.append((i, j[0], k))

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
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)#CHANGED FROM 0.01
    parser.add_argument('--load-model', type=str, default='')
    parser.add_argument('--load-epoch', type=int, default=-1)
    parser.add_argument('--model-path', type=str, default='./assets/learned_models',
                        help='pre-trained model path')
    parser.add_argument('--data-path', type=str, default='./data', help='data path')
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--save-interval', type=int, default=1)

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
        #model = PointNet2ClsMsg(num_classes=2)
        conf = config()
        conf.epochs = 64 
        conf.batch_size = 16
        conf.learning_rate = 0.001
        conf.lr_shrink_rate = 0.8
        conf.lr_min = 0.00001
        conf.regularization = 5e-4
        conf.N = 1024 # max is 2048
        conf.nCls = 2 #CHANGED FROM 40
        conf.k = 20
        conf.cuda = False #CHANGED FROM TRUE
        conf.workers = 1
        conf.print_freq = 1 #CHANGED FROM 50

        model = dgcnn(conf)
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
