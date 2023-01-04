----
Using HPUDataModule
-----------------------

``HPUDataModule`` class is a wrapper around the ``LightningDataModule`` class. It makes working with custom models easier on HPU devices.
It uses HabanaDataloader for training, testing, and validation of user-provided models. Currently, it only supports the ``Imagenet`` dataset.

Here's an example of how to use the ``HPUDataModule``:

.. code-block:: python

    import pytorch_lightning as pl
    from habana_datamodule.hpu_datamodule import HPUDataModule

    train_dir = "./path/to/train/data"
    val_dir = "./path/to/val/data"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    val_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]

    data_module = HPUDataModule(
        train_dir,
        val_dir,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=8,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    # Initialize a trainer
    trainer = pl.Trainer(devices=1, accelerator="hpu", max_epochs=1, max_steps=2)

    # Init our model
    model = RN50Module()  # Or any other model to be defined by user

    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)

A working example can be found at ``examples/hpu_datamodule_sample.py``.
For more details refer to `Habana dataloader <https://docs.habana.ai/en/latest/PyTorch_User_Guide/PyTorch_User_Guide.html#habana-data-loader>`__.
