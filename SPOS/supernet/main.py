import os
import sys
import time
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from super_model import SuperNetwork
from train import train

from config import config
import functools
print=functools.partial(print,flush=True)
import pickle
import copy
import apex

sys.path.append("../..")
from utils import *

IMAGENET_TRAINING_SET_SIZE = 1231167

parser = argparse.ArgumentParser("ImageNet")
parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.25, help='init learning rate')
parser.add_argument('--min_lr', type=float, default=5e-4, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=30, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--total_iters', type=int, default=300000, help='total iters')
parser.add_argument('--classes', type=int, default=1000, help='number of classes')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--train_dir', type=str, default='../../data/train', help='path to training dataset')
parser.add_argument('--operations_path', type=str, default='./shrunk_search_space.pt', help='shrunk search space')
args = parser.parse_args()

per_epoch_iters = IMAGENET_TRAINING_SET_SIZE // args.batch_size

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    num_gpus = torch.cuda.device_count() 
    np.random.seed(args.seed)
    args.gpu = args.local_rank % num_gpus
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    group_name = 'search_space_shrinking'
    print('gpu device = %d' % args.gpu)
    print("args = %s", args)

    torch.distributed.init_process_group(backend='nccl', init_method='env://', group_name = group_name)
    args.world_size = torch.distributed.get_world_size()
    args.batch_size = args.batch_size // args.world_size

    criterion_smooth = CrossEntropyLabelSmooth(args.classes, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    # Prepare model
    base_model = SuperNetwork().cuda(args.gpu)
    model = SuperNetwork().cuda(args.gpu)
    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    
    # Max iterations
    args.total_iters = args.epochs * per_epoch_iters
    optimizer, scheduler = get_optimizer_schedule(model, args)
    # set_trace()
    # Prepare data
    train_loader = get_train_dataloader(args.train_dir, args.batch_size, args.local_rank, args.total_iters)
    train_dataprovider = DataIterator(train_loader)

    operations = [list(range(config.op_num)) for i in range(config.layers)]
    print('operations={}'.format(operations))

    seed = args.seed
    start_iter, ops_dim, modify_initial_model = 0, 0, False
    checkpoint_tar = config.checkpoint_cache
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar,map_location={'cuda:0':'cuda:{}'.format(args.local_rank)})
        start_iter = checkpoint['iter'] + 1
        seed = checkpoint['seed']
        operations = checkpoint['operations']
        modify_initial_model = checkpoint['modify_initial_model']
        model.load_state_dict(checkpoint['state_dict'])

        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        print('{} load checkpoint..., iter = {}, operations={}'.format(now, start_iter, operations))

        # reset the scheduler
        cur_iters = (config.first_stage_epochs + (start_iter-1)*config.other_stage_epochs)*per_epoch_iters if start_iter > 0 else 0
        for _ in range(cur_iters):
            if scheduler.get_lr()[0] > args.min_lr:
                scheduler.step()
        print('resume from iters={}'.format(cur_iters))


    for i in range(start_iter, args.epochs):

        per_stage_iters = per_epoch_iters
        seed = train(train_dataprovider, optimizer, scheduler, model, criterion_smooth, operations, i, per_stage_iters, seed, args)
        if args.local_rank == 0:
            save_checkpoint({'operations': operations,
                             'iter': i,
                             'state_dict': model.state_dict(),
                             'seed': seed
                             }, config.checkpoint_cache)



if __name__ == '__main__':
  main() 