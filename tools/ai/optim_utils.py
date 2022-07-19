import torch
from .torch_utils import *

class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, warmup_epoch, max_step, momentum=0.9, nesterov=False):
        super().__init__(params, lr, weight_decay, nesterov=nesterov)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum
        self.warmup_epoch = warmup_epoch
        
        self.__initial_lr = [group['lr'] for group in self.param_groups]
    

    def step(self, closure=None):
        # max_step: 4950
        # epoch_step: 330
        if self.global_step < (self.warmup_epoch-1) * 661:

            lr_mult = (self.global_step // 661 + 1) / self.warmup_epoch

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult
        else:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

class PolyOptimizer_adam(torch.optim.Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        super().__init__(params, lr)

        # self.global_step = 0
        # self.max_step = max_step
        # self.momentum = momentum
        # self.warmup_epoch = warmup_epoch
        
        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        for i in range(len(self.param_groups)):
            self.param_groups[i]['lr'] = self.__initial_lr[i] 

        super().step(closure)


