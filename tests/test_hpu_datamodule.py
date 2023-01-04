import os
from unittest import mock

import pytest
import torch

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
from tests_pytorch.helpers.runif import RunIf
from habana_lightning_plugins.datamodule import HPUDataModule

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms

@RunIf(hpu=True)
def test_hpu_datamodule():

    data_module = HPUDataModule(num_workers=8, batch_size=32, shuffle=False, pin_memory=True)
    assert isinstance(data_module, LightningDataModule)


@RunIf(hpu=True)
def test_hpu_datamodule_shuffle():

    data_module = HPUDataModule(num_workers=8, batch_size=32, shuffle=True, pin_memory=True)

    model = BoringModel()
    trainer = Trainer(devices=1, accelerator="hpu", max_epochs=1)
    with pytest.raises(ValueError) as excinfo:
        trainer.fit(model, datamodule=data_module)

    assert excinfo.type is ValueError
    assert str(excinfo.value) == "HabanaDataLoader does not support shuffle=True"


@RunIf(hpu=True)
def test_hpu_datamodule_pin_memory():

    data_module = HPUDataModule(
        num_workers=8,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
    )

    model = BoringModel()
    trainer = Trainer(devices=1, accelerator="hpu", max_epochs=1)
    with pytest.raises(ValueError) as excinfo:
        trainer.fit(model, datamodule=data_module)

    assert excinfo.type is ValueError
    assert str(excinfo.value) == "HabanaDataLoader only supports pin_memory=True"


@RunIf(hpu=True)
def test_hpu_datamodule_num_workers():

    data_module = HPUDataModule(num_workers=4, batch_size=32, shuffle=False, pin_memory=True)

    model = BoringModel()
    trainer = Trainer(devices=1, accelerator="hpu", max_epochs=1)
    with pytest.raises(ValueError) as excinfo:
        trainer.fit(model, datamodule=data_module)

    assert excinfo.type is ValueError
    assert str(excinfo.value) == "HabanaDataLoader only supports num_workers as 8"


@RunIf(hpu=True)
def test_hpu_datamodule_unsupported_transforms():

    if not _TORCHVISION_AVAILABLE:
        return None

    model = BoringModel()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.ColorJitter()
    train_transforms = [
        transform,
        transforms.ToTensor(),
        normalize,
    ]

    data_module = HPUDataModule(
        train_transforms=train_transforms, num_workers=8, batch_size=32, shuffle=False, pin_memory=True
    )

    # Initialize a trainer
    trainer = Trainer(
        devices=1,
        accelerator="hpu",
        max_epochs=1,
        precision=32,
        max_steps=1,
        limit_test_batches=0.1,
        limit_val_batches=0.1,
    )

    with pytest.raises(ValueError) as excinfo:
        trainer.fit(model, datamodule=data_module)

    assert excinfo.type is ValueError
    assert str(excinfo.value) == f"Unsupported train transform: {str(type(transform))}"
