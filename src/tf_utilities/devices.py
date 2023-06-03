import numpy as np
import re
import subprocess
from typing import cast, TypedDict

from .lazyloading import tensorflow as tf

# Module Information -------------------------------------------------------------------------------

# The current device configuration being used.
_device_config: tuple[int, int, bool] = None

# Type Definitions ---------------------------------------------------------------------------------

GpuMemoryInfo = TypedDict("GpuMemoryInfo", {"used": int, "free": int, "total": int})

# Interface Functions ------------------------------------------------------------------------------

def gpu_memory():
    """
    Get the memory usage of all GPUs on the system.

    Implementation inspired by: https://stackoverflow.com/a/59571639
    """
    command = "nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv"
    gpu_info = subprocess.check_output(command.split()).decode('ascii').rstrip().split('\n')[1:]
    return cast(tuple[GpuMemoryInfo], tuple(({
        k: v for k, v in zip(("used", "free", "total"), map(int, re.findall(r"(\d+)", gpu)))
    } for gpu in gpu_info)))


def get_cpus(cpu_count: int = 1):
    """
    Select the given number of CPUs.
    """
    return cpu_list()[:cpu_count]


def get_gpus(gpu_count: int = 1):
    """
    Select the given number of GPUs. The selected devices are prioritized by their available memory.
    """
    indices = np.argpartition([memory["free"] for memory in gpu_memory()], -gpu_count)[-gpu_count:]
    gpus = gpu_list()
    return cast(list[tf.config.PhysicalDevice], [gpus[i] for i in indices])


def use(*, cpus: int = 1, gpus: int = 0, use_dynamic_memory=True):
    """
    Selects the specified number of CPUs and GPUs. GPU devices are prioritized by their available
    memory.
    """
    global _device_config
    if _device_config is not None and _device_config != (cpus, gpus, use_dynamic_memory):
        raise RuntimeError("The device configuration has already been set.")
    cpu_devices = get_cpus(cpus)
    gpu_devices = get_gpus(gpus) if gpus > 0 else []
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, use_dynamic_memory)
    tf.config.set_visible_devices(cpu_devices + gpu_devices)
    _device_config = (cpus, gpus, use_dynamic_memory)
    return cpu_devices + gpu_devices


def cpu_list():
    """
    Get the list of visible CPU devices.
    """
    return cast(list[tf.config.PhysicalDevice], tf.config.get_visible_devices("CPU"))


def gpu_list():
    """
    Get the list of visible GPU devices.
    """
    return cast(list[tf.config.PhysicalDevice], tf.config.get_visible_devices("GPU"))
