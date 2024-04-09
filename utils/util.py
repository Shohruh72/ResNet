import cv2
import math
import random
import numpy as np
from PIL import Image
from os import environ
from platform import system

import torch
import torch.nn.functional as F


def lr(args):
    return 0.1 * args.batch_size * args.world_size / 4096


def reduce_tensor(tensor, n):
    from torch import distributed as dist
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def setup_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def weight_decay(model, wd=1e-5):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': wd}]


@torch.no_grad()
def accuracy(outputs, targets, top_k):
    results = []
    outputs = outputs.topk(max(top_k), 1, True, True)[1].t()
    outputs = outputs.eq(targets.view(1, -1).expand_as(outputs))

    for k in top_k:
        correct = outputs[:k].reshape(-1)
        correct = correct.float().sum(0, keepdim=True)
        results.append(correct.mul_(100.0 / targets.size(0)))
    return results


class CosineLR:
    def __init__(self, lr, args, optimizer):
        self.lr = lr
        self.min = 1E-5
        self.max = 1E-4
        self.args = args
        self.warmup_epochs = 5
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.max

    def step(self, epoch, optimizer):
        epochs = self.args.epochs
        if epoch < self.warmup_epochs:
            lr = self.max + epoch * (self.lr - self.max) / self.warmup_epochs
        else:
            if epoch < epochs:
                alpha = math.pi * (epoch - (epochs * (epoch // epochs))) / epochs
                lr = self.min + 0.5 * (self.lr - self.min) * (1 + math.cos(alpha))
            else:
                lr = self.min

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def resample():
    return random.choice((Image.BILINEAR, Image.BICUBIC))


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        size = self.size
        i, j, h, w = self.params(image.size)
        image = image.crop((j, i, j + w, i + h))
        return image.resize([size, size], resample())

    @staticmethod
    def params(size):
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        for _ in range(10):
            target_area = random.uniform(*scale) * size[0] * size[1]
            aspect_ratio = math.exp(random.uniform(*(math.log(ratio[0]), math.log(ratio[1]))))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        if (size[0] / size[1]) < min(ratio):
            w = size[0]
            h = int(round(w / min(ratio)))
        elif (size[0] / size[1]) > max(ratio):
            h = size[1]
            w = int(round(h * max(ratio)))
        else:
            w = size[0]
            h = size[1]
        i = (size[1] - h) // 2
        j = (size[0] - w) // 2
        return i, j, h, w
