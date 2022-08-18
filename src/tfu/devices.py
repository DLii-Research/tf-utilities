import tensorflow as tf

def device_id(device):
    return ':'.join(device.name.split(':')[-2:])

def find_devices(dev_type: str, indices: list=None):
    dev_type = dev_type.upper()
    indices = [indices] if type(indices) == int else indices
    assert dev_type in ("GPU", "CPU")
    devices = tf.config.get_visible_devices(dev_type)
    if indices is not None:
        devices = [devices[i] for i in indices]
    assert len(devices) > 0, f"Could not find specified {dev_type}s"
    return devices

def select_cpu(index: int=0):
    cpus = find_devices("CPU", index)
    tf.config.set_visible_devices(cpus)
    return cpus

def select_gpu(indices: int=None, cpu_index=0, use_dynamic_memory=True):
    cpus = find_devices("CPU", cpu_index)
    gpus = find_devices("GPU", indices)
    tf.config.set_visible_devices(cpus + gpus)
    if use_dynamic_memory:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    return cpus, gpus
