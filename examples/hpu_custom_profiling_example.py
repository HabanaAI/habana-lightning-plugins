# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This script will generate 2 traces: one for `training_step` and one for `validation_step`. The traces can be
visualized in 2 ways:

* With Chrome:
    1. Open Chrome and copy/paste this url: `chrome://tracing/`.
    2. Once tracing opens, click on `Load` at the top-right and load one of the generated traces.
* With PyTorch Tensorboard Profiler (Instructions are here: https://github.com/pytorch/kineto/tree/master/tb_plugin)
    1. pip install tensorboard torch-tb-profiler
    2. tensorboard --logdir={FOLDER}

* To profile custom actions of interest, use the profiler context manager, similar to PyTorch Profiler.
For more information: https://pytorch-lightning.readthedocs.io/en/stable/tuning/profiler_expert.html#profile-custom-actions-of-interest

"""

import os
import torch
from torch.nn import functional as F
import warnings

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning.utilities.imports import _KINETO_AVAILABLE

if _KINETO_AVAILABLE:
    from pytorch_lightning.profilers.hpu import HPUProfiler
else:
    from pytorch_lightning.profiler.pytorch import PyTorchProfiler


class SimpleMNISTModel(LightningModule):
    def __init__(self, profiler=None):
        self.profiler = profiler
        super(SimpleMNISTModel, self).__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.layer_1(x.view(x.size(0), -1)))

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def train_custom(self, train_batch):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return loss

    def training_step(self, train_batch, batch_idx):
        """
        This portion shows how to use custom profiling with HPUProfiler.
        Alternatively, one can use the context manager approach with checks:
            if self.current_epoch == 1:
                with self.proifler.profile("action_name"):
                    return self.train_custom(train_batch)
        """
        if self.current_epoch == 1:
            self.profiler.start("validation_step")
            loss = self.train_custom(train_batch)
            self.profiler.stop("validation_step")
            return loss
        return self.train_custom(train_batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class SimpleMNISTDataModule(LightningDataModule):
    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=16, num_workers=1)


if __name__ == "__main__":
    data_module = SimpleMNISTDataModule()

    if _KINETO_AVAILABLE:
        profiler = HPUProfiler(
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            dirpath='.',    # Trainer does not set dirpath for custom profiling
        )
        accelerator = "hpu"
    else:
        profiler = PyTorchProfiler()
        accelerator = "cpu"
        warnings.warn(
            f"""_KINETO_AVAILABLE is {_KINETO_AVAILABLE}. Continuing with
                      profiler="PyTorchProfiler"
                      accelerator="{accelerator}" """
        )

    # Pass profiler info to the model
    model = SimpleMNISTModel(profiler)

    # Do not pass profiler info to the trainer.
    trainer = Trainer(
        accelerator='hpu',
        devices=1,
        max_epochs=2,
        limit_train_batches=16,
        limit_val_batches=16,
    )

    trainer.fit(model, data_module)
