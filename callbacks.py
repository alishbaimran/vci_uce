# Author: marcelroed
from typing import Mapping
import torch
from torch import Tensor
from typing import Any
import lightning as L


class LogLR(L.Callback):
    def __init__(self, interval=10):
        super().__init__()
        self.interval = interval
    
    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int) -> None:
        if trainer.global_rank == 0:
            # print(f'{batch_idx=}, {trainer.global_step=}, {self.interval=} {pl_module.lr_schedulers().get_last_lr()=}')
            if trainer.global_step % self.interval == 0 and trainer.logger is not None:
                trainer.logger.log_metrics({'learning_rate': pl_module.lr_schedulers().get_last_lr()[0]}, step=trainer.global_step)