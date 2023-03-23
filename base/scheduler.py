import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Optimizer


class GradualWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr  for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr for base_lr in self.base_lrs]

        return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        self.last_epoch = 1
        if epoch is not None:
            self.last_epoch = epoch + 2

        if self.last_epoch < self.total_epoch:
            warmup_lr = [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        elif self.last_epoch == self.total_epoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr']
        else:
            # If not using warmup, then do nothing at the beginning.
            if self.last_epoch == 1 and self.total_epoch == 0:
                pass
            else:
                self.after_scheduler.step(metrics)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class MyWarmupScheduler(object):
    def __init__(self, optimizer, lr, min_lr, best=None, mode='min', patience=5, factor=0.1, num_warmup_epoch=20, init_epoch=0, eps=1e-11, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.learning_rate = lr

        self.best = best
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.num_warmup_epoch = num_warmup_epoch
        self.init_epoch = init_epoch
        self.relative_epoch = 0
        self.last_epoch = -1
        self.last_stage_epoch = None
        self.num_bad_epochs = 0
        self.eps = eps
        self.verbose = verbose

        if best is None:
            self.best = -1e10
            if mode == "min":
                self.best = 1e10

    def is_better(self, metric):

        better = 0
        if self.mode == "min":
            if metric < self.best:
                better = 1
        else:
            if metric > self.best:
                better = 1

        return better

    def warmup_lr(self, init_lr, batch,  num_batch_warm_up):
        if self.relative_epoch < self.num_warmup_epoch:
            for params in self.optimizer.param_groups:
                params['lr'] = batch * init_lr * (self.relative_epoch + 1) / (num_batch_warm_up * self.num_warmup_epoch + 1e-100)


    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            # new_lr = max(old_lr * self.factor, self.min_lrs[i])
            new_lr = old_lr * self.factor
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

    def step(self, epoch, metrics):
        self.relative_epoch = epoch - self.init_epoch + 1

        if self.relative_epoch == self.num_warmup_epoch:
            for params in self.optimizer.param_groups:
                params['lr'] = self.learning_rate

        current = float(metrics)

        if self.is_better(current):
            self.best = current
            self.num_bad_epochs = 0
        else:
            if self.relative_epoch > self.num_warmup_epoch:
                self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.num_bad_epochs = 0