#!/usr/bin/env python3
"""
Learning Rate Schedulers for ADNI MRI Model Training

This module provides two schedulers:
  - WarmupCosineSchedule: linear warmup followed by cosine decay
  - CosineAnnealingWarmUpRestarts: cosine annealing with warmup and restarts

Usage:
    from adni_lr_scheduler import get_scheduler, WarmupCosineSchedule, CosineAnnealingWarmUpRestarts

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='warmup_cosine',
        warmup_steps=500,
        total_steps=10000,
        cycles=0.5
    )

    for step in range(total_steps):
        train_step(...)
        scheduler.step()
"""
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

def get_scheduler(
        optimizer: Optimizer,
        scheduler_type: str,
        warmup_steps: int,
        total_steps: int = None,
        first_cycle_steps: int = None,
        cycle_mult: float = 1.0,
        max_lr: float = None,
        min_lr: float = 0.0,
        cycles: float = 0.5,
        gamma: float = 1.0,
        restart_interval: int = -1
) -> _LRScheduler:
    """
    Factory to create LR scheduler.

    Args:
        optimizer: torch Optimizer
        scheduler_type: 'warmup_cosine' or 'cosine_restart'
        warmup_steps: steps for linear warmup
        total_steps: total training steps (for warmup_cosine)
        first_cycle_steps: initial cycle length (for cosine_restart)
        cycle_mult: multiplier for cycle length each restart
        max_lr: maximum LR (for cosine_restart)
        min_lr: minimum LR
        cycles: number of cosine cycles (for warmup_cosine)
        gamma: LR decay per restart
        restart_interval: override cycle length for warmup_cosine wrap-around
    Returns:
        scheduler instance
    """
    if scheduler_type == 'warmup_cosine':
        assert total_steps is not None, "total_steps required for warmup_cosine"
        return WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            t_total=total_steps,
            cycles=cycles,
            restart_interval=restart_interval
        )
    elif scheduler_type == 'cosine_restart':
        assert first_cycle_steps is not None and max_lr is not None, (
            "first_cycle_steps and max_lr required for cosine_restart"
        )
        return CosineAnnealingWarmUpRestarts(
            optimizer=optimizer,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=cycle_mult,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            gamma=gamma
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

class WarmupCosineSchedule(LambdaLR):
    """
    Linear warmup followed by cosine decay.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int,
            t_total: int,
            cycles: float = 0.5,
            last_epoch: int = -1,
            restart_interval: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.restart_interval = restart_interval
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step: int) -> float:
        if self.restart_interval > 0:
            step = step % self.restart_interval
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * 2.0 * self.cycles * progress))
        )

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            first_cycle_steps: int,
            cycle_mult: float = 1.0,
            max_lr: float = 0.1,
            min_lr: float = 0.0,
            warmup_steps: int = 0,
            gamma: float = 1.0,
            last_epoch: int = -1
    ):
        assert warmup_steps < first_cycle_steps, (
            "warmup_steps must be less than first_cycle_steps"
        )
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        super().__init__(optimizer, last_epoch)
        self._init_lr()

    def _init_lr(self):
        self.base_lrs = []
        for group in self.optimizer.param_groups:
            group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self) -> list:
        if self.step_in_cycle == -1:
            return self.base_lrs
        if self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr + (self.max_lr - base_lr) *
                (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps)
                              / (self.cur_cycle_steps - self.warmup_steps))) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch: int = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle -= self.cur_cycle_steps
                self.cur_cycle_steps = (
                        int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                        + self.warmup_steps
                )
        else:
            if epoch < self.first_cycle_steps:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
            else:
                n = int(
                    math.log(
                        (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                        self.cycle_mult
                    )
                )
                self.cycle = n
                self.cur_cycle_steps = int(
                    self.first_cycle_steps * (self.cycle_mult ** n)
                )
                self.step_in_cycle = epoch - int(
                    self.first_cycle_steps * (self.cycle_mult ** n - 1)
                    / (self.cycle_mult - 1)
                )
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = epoch
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.get_lr()[i]
