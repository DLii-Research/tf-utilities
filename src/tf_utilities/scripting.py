import argparse
import os
import numpy as np
import sys
import tensorflow as tf
from . import utils as tfu_utils

# A session object for variable reference
__session = {}

# Configuration Parsing ---------------------------------------------------------------------------

class CliArgumentFactory:
    def __init__(self, description=None):
        self.parser = argparse.ArgumentParser(description=description)
        self.job_args = []

    def argument(self, *args, **kwargs):
        return self.parser.add_argument(*args, **kwargs)

    def job_argument(self, *args, **kwargs):
        arg = self.argument(*args, **kwargs)
        self.job_args.append(arg.dest)
        return arg

    def artifact(self, arg_name, *args, required=True, **kwargs):
        group = self.parser.add_mutually_exclusive_group(required=required)
        group.add_argument(arg_name + "-path", *args, **kwargs)
        group.add_argument(arg_name + "-artifact", *args, **kwargs)

    def use_strategy(self):
        self.argument("--gpus", default=None, type=lambda x: list(map(int, x.split(','))), help="Comma separated list of integers. Example: 0,1")

    def use_training(self, epochs=1, batch_size=None, sub_batch_size=0, data_workers=1):
        self.job_argument("--initial-epoch", type=int, default=0)
        self.job_argument("--epochs", type=int, default=epochs)
        self.argument("--batch-size", type=int, required=(batch_size is None), default=batch_size)
        self.argument("--sub-batch-size", type=int, default=sub_batch_size)
        self.argument("--data-workers", type=int, default=data_workers)
        self.argument("--run-eagerly", action="store_true", default=False)
        self.argument("--use-dynamic-memory", action="store_true", default=False)

    def use_wandb(self, allow_resume=True):
        self.job_argument("--wandb-project", type=str, default=None, help="W&B project name")
        self.job_argument("--wandb-name", type=str, default=None, help="W&B run name")
        self.job_argument("--wandb-group", type=str, default=None, help="W&B group name")
        self.job_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online")
        if allow_resume:
            self.job_argument("--resume", type=str, default=None, help="W&B Job ID of existing run")

    def use_rng(self):
        self.argument("--seed", type=int, default=None, required=False, help="Random seed")

    def parse(self, argv):
        config = self.parser.parse_args(argv)
        job_config = self.__extract_job_config(config)

        # If a run ID was specified explicitly, we should only
        # keep the CLI arguments that were supplied explicitly
        # as all other values should default to the previous run.
        if hasattr(job_config, "resume") and job_config.resume is not None:
            supplied_args = self.__supplied_cli_args(argv, config)
            self.__remove_defaults_from_config(config, supplied_args)
        return job_config, config

    def __extract_job_config(self, config):
        """
        Extract the given fields from the configuration as a separate Namespace instance
        """
        extracted = {}
        config_dict = vars(config)
        for key in self.job_args:
            extracted[key] = config_dict[key]
            delattr(config, key)
        result = argparse.Namespace()
        result.__dict__.update(extracted)
        return result

    def __supplied_cli_args(self, argv, config):
        """
        Get the list of explicitly-supplied arguments from the CLI
        """
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for arg in vars(config):
            aux_parser.add_argument("--" + arg.replace('_', '-'))
        cli_args, _ = aux_parser.parse_known_args(argv[1:])
        return cli_args

    def __remove_defaults_from_config(self, config, supplied_args):
        """
        Remove default arguments from the given config
        """
        to_remove = set(vars(config).keys()) - set(vars(supplied_args).keys())
        for key in to_remove:
            delattr(config, key)

# Scripting Interface ------------------------------------------------------------------------------

def boot(job, *args, **kwargs):
    """
    Boot a job.
    """
    return job(*args, **kwargs) or 0


def configure(argv, arg_defs):
    """
    Parse the provided arguments to obtain the job configuration.
    """
    builder = CliArgumentFactory()
    for arg_def in arg_defs:
        arg_def(builder)
    return builder.parse(argv)


def init(arg_defs=None, argv=sys.argv[1:], use_wandb=True):
    """
    Initialize a new job.
    """
    if arg_defs is None:
        arg_defs = []
    elif type(arg_defs) not in (tuple, list):
        arg_defs = [arg_defs]

    # Parse the configuration
    job_config, config = configure(argv, arg_defs)

    if hasattr(config, "seed"):
        random_seed(config.seed)

    # Create the W&B instance
    __init_wandb(job_config, config, use_wandb)

    # Merge the configs
    config.__dict__.update(job_config.__dict__)
    return config

# Scripting Utilities ------------------------------------------------------------------------------

@tfu_utils.static_vars(instance=None)
def strategy(config):
    """
    Fetch a strategy instance for the corresponding config.
    """
    from . import strategy as tfu_strategy
    if strategy.instance is None:
        if config.gpus is None:
            print("Using CPU Strategy")
            strategy.instance = tfu_strategy.cpu()
        else:
            print(f"Using GPU Strategy. Selected GPUs: {config.gpus}")
            strategy.instance = tfu_strategy.gpu(config.gpus, use_dynamic_memory=config.use_dynamic_memory)
    return strategy.instance


def artifact(config, key):
    """
    Fetch the path to an artifact from the config.
    """
    key = key.replace('-', '_')
    path = getattr(config, f"{key}_path")
    if path is not None:
        return path

    import wandb
    name = getattr(config, f"{key}_artifact")
    if not is_wandb_disabled():
        artifact = wandb_run().use_artifact(name)
    else:
        artifact = wandb_api().artifact(name)

    path = None
    if "WANDB_ARTIFACTS_PATH" in os.environ and os.environ["WANDB_ARTIFACTS_PATH"] is not None:
        path = os.path.join(os.environ["WANDB_ARTIFACTS_PATH"], name)
    return artifact.download(path)


def initial_epoch(config):
    """
    Get the initial epoch for the run.
    """
    if not is_using_wandb():
        return config.initial_epoch
    run = __session
    if config.initial_epoch > 0 and wandb_run().step != config.initial_epoch:
        print("WARNING: Supplied initial epoch will be ignored while using W&B.")
    return wandb_run().step


def is_resumed():
    """
    Determine if this is a resumed job.
    """
    return __session["is_resumed"]


def restore(name, run_path=None, replace=False, root=None):
    """
    Restore the specified file from a previous run.
    """
    if not is_using_wandb():
        return name
    import wandb
    return wandb.restore(name, run_path, replace, root)


def restore_dir(name, run_path=None, replace=False, root=None):
    """
    Restore (recursively) a the given directory from a previous run.
    """
    if not is_using_wandb():
        return name
    run_path = run_path if run_path is not None else wandb_run().path
    run = wandb_api().run(run_path)
    for f in filter(lambda f: f.name.startswith(name), run.files()):
        return restore(name, run_path, replace, root)
    return os.path.join(wandb_run().dir, name)


def run_safely(fn, *args, **kwargs):
    """
    Run a function with keyboard interrupt protection.
    """
    try:
        return fn(*args, **kwargs)
    except KeyboardInterrupt:
        return None


def cwd():
    """
    Get the current working directory. If W&B is enabled, this is the current run directory.
    """
    if not is_using_wandb():
        return os.getcwd()
    return wandb_run().dir


def path_to(paths):
    """
    Prefix the given paths with the current run directory.
    This should be used for all run-specific writes performed in a job.
    """
    if type(paths) is str:
        return os.path.join(cwd(), paths)
    d = cwd()
    return [os.path.join(d, p) for p in paths]


def random_seed(seed):
    """
    Set the random seed for Python, numpy, and Tensorflow.
    """
    if seed is None:
        return
    __session["seed"] = seed
    __session["next_seed"] = seed
    tf.keras.utils.set_random_seed(seed)


def rng():
    """
    Fetch a new numpy RNG instance.
    """
    seed = None
    if "seed" in __session:
        __session["next_seed"] += 1
        seed = __session["next_seed"]
    return np.random.default_rng(seed)


# Weights and Biases  ------------------------------------------------------------------------------

def __init_wandb(job_config, config, use_wandb=True):
    """
    Initialize the W&B instance.
    """
    if not use_wandb:
        return None
    __session["is_resumed"] = job_config.resume is not None
    if not hasattr(job_config, "wandb_project"):
        return None
    if "WANDB_DISABLED" in os.environ and tfu_utils.str_to_bool(os.environ["WANDB_DISABLED"]):
        print("WARNING: Weights and Biases is currently disabled in the environment.")
        return None

    import wandb

    # Run-resume
    if hasattr(job_config, "resume") and job_config.resume is not None:
        job_id = job_config.resume
    else:
        job_id = wandb.util.generate_id()

    __session["run"] = wandb.init(
        id=job_id,
        project=job_config.wandb_project,
        name=job_config.wandb_name,
        group=job_config.wandb_group,
        mode=job_config.wandb_mode,
        config=config,
        resume=bool(job_config.resume))

def log_artifact(name, paths, type, description=None, metadata=None, incremental=None, use_as=None):
    """
    Log an artifact to W&B.
    """
    if not is_using_wandb():
        return
    import wandb
    if isinstance(paths, str):
        paths = [paths]
    artifact = wandb.Artifact(name, type, description, metadata, incremental, use_as)
    for path in paths:
        if os.path.isdir(path):
            print("Adding directory:", path)
            artifact.add_dir(path)
        else:
            print("Adding file:", path)
            artifact.add_file(path)
    print("Logging artifact:", artifact)
    wandb.log_artifact(artifact)


def is_using_wandb():
    """
    Determine if W&B is being used for this job.
    """
    return "run" in __session


def is_wandb_disabled():
    """
    Determine if W&B is disabled for this job.
    """
    if not is_using_wandb():
        return True
    return wandb_run().disabled


@tfu_utils.static_vars(instance=None)
def wandb_api():
    """
    Fetch a W&B public API instance.
    """
    if wandb_api.instance is None:
        import wandb
        wandb_api.instance = wandb.Api()
    return wandb_api.instance


def wandb_run():
    """
    Get the current W&B run instance.
    """
    return __session["run"]


def wandb_callback(*args, save_model_as_artifact=False, **kwargs):
    """
    Create a WandbCallback instance with the provided arguments if W&B is being used.
    """
    if not is_using_wandb():
        return None
    import wandb
    callback = wandb.keras.WandbCallback(*args, **kwargs)
    callback.save_model_as_artifact = save_model_as_artifact
    return callback
