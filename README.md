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


# HPUProfiler

HPUProfiler is an lightning implementation of pytorch profiler for HPU devices. It aids to get the profiling summary of PyTorch functions. 
It subclasses PyTorch Lightning's [PyTorch profiler](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.profilers.PyTorchProfiler.html#pytorch_lightning.profilers.PyTorchProfiler).

## Installation

It comes as part of Habana-Lightning-Plugins package. 

Use [pip](https://pip.pypa.io/en/stable/) to install the plugins package.

```bash
python -um pip install Habana-Lightning-Plugins
```

## Usage

### Default profiling
For auto profiling, create a HPUProfiler instance and pass it to trainer.
At the end of profiler.fit(), it will generate a json trace for the run.
In case accelerator = "hpu" is not used with HPUProfiler, then it will dump only cpu traces, similar to PyTorchProfiler.

``` python
# Import profiler
from habana_lightning_plugins.profiler import HPUProfiler

# Create profiler object
profiler = HPUProfiler()
accelerator = "hpu"

# Pass profiler to the trainer
    trainer = Trainer(
        profiler=profiler,
        accelerator=accelerator,
    )
```

### Distributed profiling

To profile a distributed model, use the HPUProfiler with the filename argument which will save a report per rank.

``` python
from habana_lightning_plugins.profiler import HPUProfiler

profiler = HPUProfiler(filename="perf-logs")
trainer = Trainer(profiler=profiler, accelerator="hpu")

```
### Custom Profiling
To [profile custom actions of interest](https://pytorch-lightning.readthedocs.io/en/stable/tuning/profiler_expert.html#profile-custom-actions-of-interest), reference a profiler in the LightningModule.

``` python
from habana_lightning_plugins.profiler import HPUProfiler

# Reference profiler in LightningModule
class MyModel(LightningModule):
    def __init__(self, profiler=None):
        self.profiler = profiler

# To profile in any part of your code, use the self.profiler.profile() function
    def custom_processing_step_basic(self, data):
        with self.profiler.profile("my_custom_action"):
            ...
        return data

# Alternatively, use self.profiler.start("my_custom_action")
# and self.profiler.stop("my_custom_action") functions
# to enclose the part of code to be profiled.
    def custom_processing_step_granular(self, data):
        self.profiler.start("my_custom_action") 
            ...
        self.profiler.stop("my_custom_action")
        return data

# Pass profiler instance to LightningModule
profiler = HPUProfiler()
model = MyModel(profiler)
trainer = Trainer(profiler=profiler, accelerator="hpu")
```
For more details on profiler, refer to [PyTorchProfiler](https://pytorch-lightning.readthedocs.io/en/stable/tuning/profiler_intermediate.html)

## Visualize profiled operations
Profiler will dump traces in json format. The traces can be
visualized in 2 ways:

### 1. With PyTorch Tensorboard Profiler (Instructions are here: https://github.com/pytorch/kineto/tree/master/tb_plugin)
``` python
# Install tensorbaord
python -um pip install tensorboard torch-tb-profiler

# Start the TensorBoard server (default at port 6006):
tensorboard --logdir ./tensorboard --port 6006

# Now open the following url on your browser
http://localhost:6006/#profile
```

### 2. With Chrome:
    1. Open Chrome and copy/paste this URL: `chrome://tracing/`.
    2. Once tracing opens, click on `Load` at the top-right and load one of the generated traces.

## Limitations

a. When using the HPUProfiler, wall clock time will not be representative of the true wall clock time. This is due to forcing profiled operations to be measured synchronously, when many HPU ops happen asynchronously. It is recommended to use this Profiler to find bottlenecks/breakdowns, however for end to end wall clock time use the SimpleProfiler.

b. HPUProfiler.summary() is not supported

c. Passing profiler name as string "hpu" to the trainer is not supported.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
