import argparse
import tqdm
import csv
import os
import copy
import torch
import torch.distributed as dist
from torch.utils import data
from torchvision import transforms

from nets import nn
from utils import util
from utils.dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def train(args):
    util.setup_seed()
    util.setup_multi_processes()
    scaler = torch.cuda.amp.GradScaler()
    model = nn.resnet(args.model_name, args.num_cls).to(device)
    optimizer = torch.optim.SGD(util.weight_decay(model), lr=util.lr(args))
    criterion = util.LabelSmoothingCrossEntropy(0.1).to(device)
    scheduler = util.CosineLR(util.lr(args), args, optimizer)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, [args.rank])

    sampler = None
    dataset = Dataset(os.path.join(args.data_dir, 'train'),
                      transforms.Compose([util.Resize(size=args.input_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), normalize]))

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, not args.distributed,
                             sampler=sampler, num_workers=8, pin_memory=True)

    with open('./outputs/weights/log.csv', 'w') as f:
        best = 0
        writer = csv.DictWriter(f, fieldnames=['epoch', 'acc@1', 'acc@5', 'train_loss', 'val_loss'])
        writer.writeheader()

        for epoch in range(args.epochs):
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                p_bar = tqdm.tqdm(loader, total=len(loader))
            losses_m = util.AverageMeter()
            model.train()
            for images, labels in p_bar:
                images = images.to(device)
                labels = labels.to(device)

                with torch.cuda.amp.autocast():
                    output = model(images)
                loss = criterion(output, labels)
                optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize(args.rank)

                if args.distributed:
                    loss = util.reduce_tensor(loss.data, args.world_size)

                losses_m.update(loss.item(), images.size(0))

                if args.rank == 0:
                    p_bar.update(1)
                    gpus = '%.4gG' % (torch.cuda.memory_reserved() / 1E9)
                    p_bar.set_description(
                        ('%10s' * 2 + '%10.3g') % ('%g/%g' % (epoch + 1, args.epochs), gpus, losses_m.avg))
            if args.rank == 0:
                p_bar.close()
            scheduler.step(epoch + 1, optimizer)

            if args.rank == 0:
                last = validate(args, model)
                writer.writerow({'acc@1': str(f'{last[1]:.3f}'),
                                 'acc@5': str(f'{last[2]:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3),
                                 'val_loss': str(f'{last[0]:.3f}'),
                                 'train_loss': str(f'{losses_m.avg:.3f}')})
                f.flush()

                state = {'model': copy.deepcopy(model.module if args.distributed else model)}
                torch.save(state, './outputs/weights/last.pt')
                if last[1] > best:
                    torch.save(state, './outputs/weights/best.pt')
                del state
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()


def validate(args, model=None):
    if model is None:
        model = torch.load('./outputs/weights/best.pt', 'cuda')['model'].float()
    model.eval().to(device)

    dataset = Dataset(os.path.join(args.data_dir, 'val'),
                      transforms.Compose([transforms.Resize(args.input_size + 32),
                                          transforms.CenterCrop(args.input_size),
                                          transforms.ToTensor(), normalize]))
    loader = data.DataLoader(dataset, 32, num_workers=8, pin_memory=True)

    top1 = util.AverageMeter()
    top5 = util.AverageMeter()
    m_loss = util.AverageMeter()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        for samples, targets in tqdm.tqdm(loader, desc='Validation', leave=False):
            # for samples, targets in tqdm.tqdm(loader, ('%10s' * 3) % ('acc@1', 'acc@5', 'loss')):
            samples, targets = samples.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(samples)
            torch.cuda.synchronize()
            acc1, acc5 = util.accuracy(outputs, targets, (1, 5))
            loss = criterion(outputs, targets)

            top1.update(acc1.item(), samples.size(0))
            top5.update(acc5.item(), samples.size(0))
            m_loss.update(loss.item(), samples.size(0))

    acc1, acc5 = top1.avg, top5.avg
    print('%10.3g' * 3 % (acc1, acc5, m_loss.avg))
    if model is None:
        torch.cuda.empty_cache()
    else:
        return m_loss.avg, acc1, acc5


# def profile(args):
#     import thop
#     model = nn.resnet(args.model_name, args.num_cls).to(device)
#     shape = (1, 3, args.input_size, args.input_size)
#
#     model.eval()
#     model(torch.zeros(shape))
#
#     x = torch.empty(shape)
#     flops, num_params = thop.profile(copy.copy(model), inputs=[x], verbose=False)
#     flops, num_params = thop.clever_format(nums=[flops, num_params], format="%.3f")
#
#     if args.local_rank == 0:
#         print(f'Number of parameters: {num_params}')
#         print(f'Number of FLOPs: {flops}')
#
#     if args.benchmark:
#         # Latency
#         model = nn.resnet(args.model_name, args.num_cls).to(device)
#         model.eval()
#
#         x = torch.zeros(shape)
#         for i in range(10):
#             model(x)
#
#         total = 0
#         import time
#         for i in range(1_000):
#             start = time.perf_counter()
#             with torch.no_grad():
#                 model(x)
#             total += time.perf_counter() - start
#
#         print(f"Latency: {total / 1_000 * 1_000:.3f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='18')
    parser.add_argument('--data-dir', type=str, default='/media/yusuf/data/Datasets/imagenet')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--num-cls', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    # args.world_size = torch.cuda.device_count()
    # args.distributed = args.world_size > 1
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    # profile(args)

    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == '__main__':
    main()
