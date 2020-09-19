from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision.transforms as T

import data_manager
from dataset_loader import ImageDataset_seg, ImageDataset
import spatial_transforms as ST
import models
from losses import CrossEntropyLabelSmooth, TripletLoss, MaskLoss, DeepSupervision
from utils.iotools import save_checkpoint, check_isfile
from utils.avgmeter import AverageMeter
from utils.logger import Logger
from utils.torchtools import count_num_param
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
from optimizers import init_optim


parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='market1501_seg',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0,
                    help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=64, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int,
                    help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.00035, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[20, 40], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3,
                    help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='IAUnet', choices=models.get_names())
parser.add_argument('--save-dir', type=str, default='./result/market/IAUnet')
# spatial attention loss
parser.add_argument('--mode', type=str, default='ce',
                    help='mask loss mode, in {l1, l2, ce}')
parser.add_argument('--alpha', type=float, default=0.5,
                    help="the weight for the spatial attention loss")
# data process
parser.add_argument('--flip_cnt', default=1, type=int) 
# Miscs
parser.add_argument('--distance', type=str, default='consine')
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--eval-step', type=int, default=10,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--gpu-devices', default='2, 3', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'), mode='a')
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_imgreid_dataset(
        name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    transform_train = ST.Compose([
        ST.Scale((args.height, args.width), interpolation=3),
        ST.RandomHorizontalFlip(),
        ST.ToTensor(),
        ST.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ST.RandomErasing(0.5),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        ImageDataset_seg(dataset.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch,  num_classes=dataset.num_train_pids) 
    print(model)
    print("Model size: {:.3f} M".format(count_num_param(model)))
    
    criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=args.margin, distance=args.distance)
    criterion_mask = MaskLoss(mode=args.mode) 
    
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

    if args.resume:
        if check_isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            rank1 = checkpoint['rank1']
            print("Loaded checkpoint from '{}'".format(args.resume))
            print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, rank1))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(args.start_epoch, args.max_epoch):
        scheduler.step()

        start_train_time = time.time()
        train(epoch, model, criterion_xent, criterion_htri, criterion_mask, optimizer, trainloader)
        train_time += round(time.time() - start_train_time)
        
        if (epoch + 1) % args.eval_step == 0 or epoch == 0:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))


def train(epoch, model, criterion_xent, criterion_htri, criterion_mask, optimizer, trainloader, use_gpu=True):
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    mask_losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    end = time.time()
    for batch_idx, (sequences, pids, _) in enumerate(trainloader):
        if use_gpu:
            sequences, pids = sequences.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        imgs = sequences[:, :3] 
        head_masks = sequences[:, 3:4] 
        upper_masks = sequences[:, 4:5] 
        lower_masks = sequences[:, 5:6] 
        shoes_masks = sequences[:, 6:7] 
        
        outputs, features, a_head, a_upper, a_lower, a_shoes = model(imgs)
        _, preds = torch.max(outputs.data, 1)
        xent_loss = criterion_xent(outputs, pids)
        htri_loss = criterion_htri(features, pids)
        loss = xent_loss + htri_loss

        head_loss = DeepSupervision(criterion_mask, a_head, head_masks)
        upper_loss = DeepSupervision(criterion_mask, a_upper, upper_masks)
        lower_loss = DeepSupervision(criterion_mask, a_lower, lower_masks)
        shoes_loss = DeepSupervision(criterion_mask, a_shoes, shoes_masks)
        mask_loss = (head_loss + upper_loss + lower_loss + shoes_loss) / 4.0

        total_loss = loss + args.alpha * mask_loss

        # backward + optimize
        total_loss.backward()
        optimizer.step()

        # statistics
        accs.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        xent_losses.update(xent_loss.item(), pids.size(0))
        htri_losses.update(htri_loss.item(), pids.size(0))
        mask_losses.update(mask_loss.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'xentLoss:{xent_loss.avg:.4f} '
          'triLoss:{tri_loss.avg:.4f} '
          'MaskLoss:{mask_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time,
           data_time=data_time, xent_loss=xent_losses,
           tri_loss=htri_losses, mask_loss=mask_losses, acc=accs))


def fliplr(img, use_gpu):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()
    if use_gpu: inv_idx = inv_idx.cuda()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            end = time.time()

            n, c, h, w = imgs.size()
            features = torch.FloatTensor(n, model.module.feat_dim).zero_()
            for i in range(args.flip_cnt):
                if (i==1):
                    imgs = fliplr(imgs, use_gpu)
                f = model(imgs)[1]
                f = f.data.cpu()
                features = features + f

            batch_time.update(time.time() - end)

            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):

            end = time.time()

            n, c, h, w = imgs.size()
            features = torch.FloatTensor(n, model.module.feat_dim).zero_()
            for i in range(args.flip_cnt):
                if (i==1):
                    imgs = fliplr(imgs, use_gpu)
                if use_gpu: imgs = imgs.cuda()
                f = model(imgs)[1]
                f = f.data.cpu()
                features = features + f

            batch_time.update(time.time() - end)

            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    
    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))
    if args.distance == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
    else:
        q_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
        g_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
        qf = qf.div(q_norm.expand_as(qf))
        gf = gf.div(g_norm.expand_as(gf))
        distmat = - torch.mm(qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]


if __name__ == '__main__':
    main()
