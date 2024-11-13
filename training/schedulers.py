import numpy as np
import bisect

class LinearCosineScheduler:
    def __init__(self, lr, total_steps, start_scale=0.01, end_scale=0.001, warmup_steps=10):
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.total_steps = total_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.lr * self.start_scale + ((self.lr - self.start_scale * self.lr) * step / self.warmup_steps)
        else:
            coeff = (1 + np.cos(np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))) / 2
            return self.lr * self.end_scale + (self.lr - self.lr * self.end_scale) * coeff

    def visualize(self):
        import matplotlib.pyplot as plt
        x = np.arange(self.total_steps)
        y = [self(i) for i in x]
        plt.plot(x, y)

class EpochBasedScheduler:
    def __init__(self, initial_lr, lr_schedule):
        self.initial_lr = initial_lr
        self.lr_schedule = sorted(lr_schedule)
        self.epochs = [epoch for epoch, _ in self.lr_schedule]
        self.lrs = [lr for _, lr in self.lr_schedule]

    def __call__(self, epoch):
        pos = bisect.bisect_right(self.epochs, epoch) - 1
        if pos < 0:
            return self.initial_lr
        else:
            return self.lrs[pos]

    def visualize(self, total_steps=100):
        import matplotlib.pyplot as plt
        x = np.arange(total_steps)
        y = [self(i) for i in x]
        plt.plot(x, y) 


class WarmupStepWiseScheduler:
    def __init__(self, initial_lr, lr_schedule, warmup_steps, warmup_lr):
        """
        Initializes the scheduler with a warm-up phase followed by step-wise changes.

        Args:
            initial_lr (float): The starting learning rate.
            lr_schedule (list of tuples): A list of (epoch, learning_rate) tuples, sorted by epoch,
                                          indicating step-wise changes after the warm-up.
            warmup_steps (int): The number of steps for the warm-up phase.
            warmup_lr (float): The learning rate at the end of the warm-up phase. 
        """
        self.initial_lr = initial_lr
        self.lr_schedule = sorted(lr_schedule)
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.epochs = [epoch for epoch, _ in self.lr_schedule]
        self.lrs = [lr for _, lr in self.lr_schedule]

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            # Linear warm-up
            answer =  self.initial_lr + ((self.warmup_lr - self.initial_lr) * epoch / (self.warmup_steps))
        else:
            # Step-wise changes after warm-up
            pos = bisect.bisect_right(self.epochs, epoch) - 1
            if pos < 0:
                answer = self.warmup_lr
            else:
                answer =  self.lrs[pos]
        return answer

    def visualize(self, total_steps=100):
        import matplotlib.pyplot as plt
        x = np.arange(total_steps)
        y = [self(i) for i in x]
        plt.plot(x, y)