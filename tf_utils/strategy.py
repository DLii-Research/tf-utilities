import tensorflow as tf
import . devices

def create_strategy(devices):
    ids = [device_id(device) for device in devices]
    if len(devices) == 1:
        return tf.distribute.OneDeviceStrategy(ids[0])
    return tf.distribute.MirroredStrategy(ids)

def cpu(index: int=0):
    cpus = devices.select_cpu(index)
    return create_strategy(cpus)

def gpu(indices: int=None, cpu_index=0):
    gpus = devices.select_gpu(index)
    tf.config.set_visible_devices(gpus)
    return create_strategy(gpus)