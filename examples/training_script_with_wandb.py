import tensorflow.keras as keras
import tfu
import tfu.scripting as ts
import sys


def define_arguments(cli):
    """
    Define the command line arguments for the training script.

    The given cli argument is a small wrapper for argparse.
    If direct access is needed to the argparse instance,
    it can be referenced using `cli.parser`.
    """

    # Include standard strategy arguments:
    # * --gpus - comma separated list of integers
    cli.use_strategy()

    # Dataset ---------------------------------------------------------

    # Dataset artifact
    # This will provide --dataset-path and --dataset-artifact CLI args.
    # cli.artifact("--dataset", type=str, required=True)

    # Architecture Settings -------------------------------------------

    cli.argument("--activation-fn", type=str, default="relu")

    # Hidden layer dimensions
    cli.argument("--hidden-dim", type=int, default=64)

    # Training settings -----------------------------------------------

    # Include standard training CLI arguments
    # * --batch-size
    # * --epochs               - The number of epochs to train
    # * --initial-epoch        - The epoch to start on
    # * --sub-batch-size       - sub-batch size for training with sub-batching
    # * --data-workers"        - data workers for data generation
    # * [--run-eagerly]        - enable eager execution
    # * [--use-dynamic-memory] - dynamic memory allocation
    # Specified arguments will serve as the defaults
    cli.use_training(epochs=20, batch_size=32)

    # Include an optional seed argument
    cli.argument("--seed", type=int, default=None)

    # Allow specifying the optimizer
    cli.argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam")

    # Learning rate for the model
    cli.argument("--lr", type=float, default=1e-3)

    # Logging ---------------------------------------------------------

    # Save the model to disk
    cli.argument("--save-to", type=str, default=None)

    # Publish the saved model to a W&B artifact
    cli.argument("--log-artifact", type=str, default=None)


def load_dataset(config):
    """
    Load the dataset for training.
    """
    # A Wandb dataset may be pulled using:
    # datadir = ts.artifact(config, "dataset")

    # Load the MNIST dataset using Keras
    (train_x, train_y), _ = keras.datasets.mnist.load_data()

    # Normalize it
    train_x = train_x / 255.0
    return (train_x, train_y)


def load_model(model_path):
    """
    Load the model from disk if resuming.
    """
    return keras.models.load_model(model_path)


def create_model(config):
    """
    Create a compiled model.
    """
    print("Creating model...")

    # Fetch the optimizer class reference
    if config.optimizer.lower() == "adam":
        OptimizerClass = keras.optimizers.Adam
    elif config.optimizer.lower() == "sgd":
        OptimizerClass = keras.optimizers.SGD


    # Create a model
    model = keras.Sequential([
        keras.layers.Input((28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(config.hidden_dim, activation=config.activation_fn),
        keras.layers.Dense(config.hidden_dim, activation=config.activation_fn),
        keras.layers.Dense(10)
    ])
    # Compile the model
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=OptimizerClass(config.lr),
        metrics=keras.metrics.SparseCategoricalAccuracy(),
        run_eagerly=config.run_eagerly # Useful for debugging
    )
    return model


def create_callbacks(config):
    """
    Create the callbacks to be used for training.
    """
    print("Creating callbacks...")
    callbacks = []
    # If using W&B, include a WandbCallback instance
    if ts.is_using_wandb():
        callbacks.append(ts.wandb_callback(save_weights_only=True))
    return callbacks


def train(config, model_path):
    """
    Train the model.
    """
    # Specify a training strategy
    with ts.strategy(config).scope():
        # Load the dataset
        train_x, train_y = load_dataset(config)

        # Create or load the model
        if model_path is not None:
            model = load_model(model_path)
        else:
            model = create_model(config)

        # Create the callbacks for training
        callbacks = create_callbacks(config)

        # Train the model with keyboard-interrupt protection
        ts.run_safely(
            model.fit,
            train_x,
            train_y,
            validation_split=0.2,
            initial_epoch=ts.initial_epoch(config),
            epochs=config.epochs,
            callbacks=callbacks,
            use_multiprocessing=(config.data_workers > 1),
            workers=config.data_workers)

        # Save the model
        if config.save_to:
            ts.save_model(model, ts.path_to(config.save_to))

    return model


def main():
    """
    Run the job
    """
    # Initialize the script and obtain the config object
    config = ts.init(define_arguments)

    # Set the random seed
    ts.random_seed(config.seed)

    # If this is a resumed run, we need to fetch the latest model run
    model_path = None
    if ts.is_resumed():
        print("Restoring previous model...")
        model_path = ts.restore_dir(config.save_to)

    # Train the model if necessary
    if ts.initial_epoch(config) < config.epochs:
        train(config, model_path)
    else:
        print("Skipping training...")

    # Upload an artifact of the model if requested
    if config.log_artifact:
        print("Logging artifact to", config.save_to)
        assert bool(config.save_to)
        ts.log_artifact(config.log_artifact, ts.path_to(config.save_to), type="model")


# Only run the job if this script was executed directly
if __name__ == "__main__":
    sys.exit(ts.boot(main))