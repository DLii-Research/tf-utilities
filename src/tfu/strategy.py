import tensorflow as tf
from . import devices

def create_strategy(cpus=[], gpus=[]):
    ids = [devices.device_id(device) for device in cpus + gpus]
    if len(gpus) < 2:
        return tf.distribute.OneDeviceStrategy(ids[-1])
    return tf.distribute.MirroredStrategy(ids)

def cpu(index: int=0):
    cpus = devices.select_cpu(index)
    return create_strategy(cpus)

def gpu(indices: int=None, cpu_index=0, use_dynamic_memory=True):
    cpus, gpus = devices.select_gpu(indices, cpu_index, use_dynamic_memory)
    return create_strategy(cpus, gpus)
