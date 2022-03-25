import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


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