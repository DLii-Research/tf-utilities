# tfu

A small library of many useful utilities for Tensorflow training workflows.

## Filtering Visible Devices

In some instances, you may want to filter what devices are visible to Tensorflow. For example, you may have multiple GPUS, but would prefer to use only a single GPU instance when training your model. `tf_utils` includes some convenience methods to assist with this.

Selecting a single GPU:

```python
visible_devices = tfu.devices.select_gpu(0)
```


Selecting a subset of GPUs:

```python
visible_devices = tfu.devices.select_gpu([0, 1])
```

## Training Strategies

When a training strategy is needed, in particular `OneDeviceStrategy` or `MirroredDeviceStrategy`, `tf_utils` provides some convenience functions to create these strategies automatically and filter out unused devices. Some examples are provided below.

Use the CPU only via a OneDeviceStrategy:

```python
strategy = tfu.strategy.cpu(0)
```

Use a single GPU via the OneDeviceStrategy:

```python
strategy = tfu.strategy.gpu(0)
```

Use multiple GPUs via the MirroredDeviceStrategy:

```python
strategy = tfu.strategy.gpu([0, 1])
```

## Dynamic Memory Growth

Tensorflow has the ability to use dynamic memory allocation, rather than allocating all of the memory on the GPU at once. Enabling dynamic memory allocation allows you to not only monitor memory usage of your models during training, but it also grants you the ability to run multiple models on a single GPU instance. All device selection and strategy functions within `tf_utils` support this feature via the `use_dynamic_memory` flag.

Enabling dynamic memory growth via device selection:

```python
visible_devices = tfu.devices.select_gpu(0, use_dynamic_memory=True)
```

Enabling dynamic memory growth via strategy creation:

```python
strategy = tfu.strategy.gpu(0, use_dynamic_memory=True)
```