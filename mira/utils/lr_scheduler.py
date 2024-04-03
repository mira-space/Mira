import numpy as np
import torch
import torch.optim as optim


def build_LR_scheduler(optimizer, scheduler_name, lr_decay_ratio, max_epochs, start_epoch=0):
    #print("-LR scheduler:%s"%scheduler_name)
    if scheduler_name == 'LambdaLR':
        decay_ratio = lr_decay_ratio
        decay_epochs = max_epochs
        polynomial_decay = lambda epoch: 1 + (decay_ratio - 1) * ((epoch+start_epoch)/decay_epochs)\
            if (epoch+start_epoch) < decay_epochs else decay_ratio
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=polynomial_decay)
    elif scheduler_name == 'CosineAnnealingLR':
        last_epoch = -1 if start_epoch == 0 else start_epoch
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, last_epoch=last_epoch)
    elif scheduler_name == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=0.01, patience=5)
    else:
        raise NotImplementedError
    return lr_scheduler


class LambdaLRScheduler:
    # target: torch.optim.lr_scheduler.LambdaLR
    def __init__(self, start_step, final_decay_ratio, decay_steps):
        self.final_decay_ratio = final_decay_ratio
        self.decay_steps = decay_steps
        self.start_step = start_step
    
    def schedule(self, step):
        if step + self.start_step < self.decay_steps:
            return 1.0 + (self.final_decay_ratio - 1) * ((step+self.start_step) / self.decay_steps)
        else:
            return self.final_decay_ratio
    
    def __call__(self, step):
        return self.scheduler(step)


class CosineAnnealingLRScheduler:
    # target: torch.optim.lr_scheduler.CosineAnnealingLR
    def __init__(self, start_step, decay_steps):
        self.decay_steps = decay_steps
        self.start_step = start_step
        
    def __call__(self, step):
        pass



## From Stable Diffusion: https://github.com/CompVis/latent-diffusion =======================================

class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __call__(self, n, **kwargs):
        return self.schedule(n,**kwargs)


class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """
    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                       f"current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (
                    1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                       f"current cycle {cycle}")

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f

