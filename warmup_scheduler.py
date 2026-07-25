"""Learning-rate warmup scheduler."""

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    """Warm up the learning rate before handing control to another scheduler.

    When ``multiplier == 1``, learning rates increase linearly from zero to the
    optimizer's base learning rates over ``total_epoch`` epochs.
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        if multiplier < 1.0:
            raise ValueError("multiplier must be greater than or equal to 1")
        self.multiplier = float(multiplier)
        self.total_epoch = int(total_epoch)
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler is None:
                return [base_lr * self.multiplier for base_lr in self.base_lrs]
            if not self.finished:
                self.after_scheduler.base_lrs = [
                    base_lr * self.multiplier for base_lr in self.base_lrs
                ]
                self.finished = True
            return self.after_scheduler.get_last_lr()

        if self.multiplier == 1.0:
            scale = float(self.last_epoch) / max(1, self.total_epoch)
        else:
            scale = 1.0 + (self.multiplier - 1.0) * self.last_epoch / max(1, self.total_epoch)
        return [base_lr * scale for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        epoch = self.last_epoch + 1 if epoch is None else epoch
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            if self.multiplier == 1.0:
                scale = float(self.last_epoch) / max(1, self.total_epoch)
            else:
                scale = 1.0 + (self.multiplier - 1.0) * self.last_epoch / max(1, self.total_epoch)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group["lr"] = base_lr * scale
        elif self.after_scheduler is not None:
            self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if isinstance(self.after_scheduler, ReduceLROnPlateau):
            self.step_ReduceLROnPlateau(metrics, epoch)
            return

        if self.finished and self.after_scheduler is not None:
            if epoch is None:
                self.after_scheduler.step()
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            super().step(epoch)

    def state_dict(self):
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "after_scheduler")
        }
        state["after_scheduler_state"] = (
            self.after_scheduler.state_dict() if self.after_scheduler is not None else None
        )
        return state

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        after_state = state.pop("after_scheduler_state", None)
        self.__dict__.update(state)
        if self.after_scheduler is not None and after_state is not None:
            self.after_scheduler.load_state_dict(after_state)
