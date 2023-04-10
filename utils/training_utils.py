import numpy as np
from typing import List, Optional, Any
from tf2_gnn.layers import get_known_message_passing_classes
from tf2_gnn.cli_utils.task_utils import get_known_tasks
from tensorflow.python.training.tracking import data_structures as tf_data_structures
from tf2_gnn.cli_utils.model_utils import save_model, load_weights_verbosely, get_model_and_dataset
from tf2_gnn.data import DataFold

import time
SMALL_NUMBER = 1e-7


def get_class_balancing_weights(class_counts: List[int], class_weight_factor: float) -> np.ndarray:
    class_counts = np.array(class_counts, dtype=np.float32)
    total_count = np.sum(class_counts)

    class_occ_weights = (total_count / (class_counts + SMALL_NUMBER)) / class_counts.shape[0]

    class_occ_weights = np.clip(class_occ_weights, a_min=0.1, a_max=10)
    class_std_weights = np.ones(class_occ_weights.shape)

    class_weights = (
        class_weight_factor * class_occ_weights + (1 - class_weight_factor) * class_std_weights
    ).astype(np.float32)

    return class_weights

def make_run_id(model_name: str, task_name: str, run_name: Optional[str] = None) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    if run_name is not None:
        return run_name
    else:
        return "%s_%s__%s" % (model_name, task_name, time.strftime("%Y-%m-%d_%H-%M-%S"))

def log_line(log_file: str, msg: str):
    with open(log_file, "a") as log_fh:
        log_fh.write(msg + "\n")
    # print(msg)

def unwrap_tf_tracked_data(data: Any) -> Any:
    if isinstance(data, (tf_data_structures.ListWrapper, list)):
        return [unwrap_tf_tracked_data(e) for e in data]
    elif isinstance(data, (tf_data_structures._DictWrapper, dict)):
        return {k: unwrap_tf_tracked_data(v) for k, v in data.items()}
    else:
        return data

def get_train_cli_arg_parser(task: str, data_path: str, default_model_type: Optional[str] = None):
    """
    Get an argparse argument parser object with common options for training
    GNN-based models.

    Args:
        default_model_type: If provided, the model type is downgraded from a
            positional parameter on the command line to an option with the
            given default value.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train a GNN model.")
    # We use a somewhat horrible trick to support both
    #  train.py --model MODEL --task TASK --data_path DATA_PATH
    # as well as
    #  train.py model task data_path
    # The former is useful because of limitations in AzureML; the latter is nicer to type.

    model_param_name, task_param_name, data_path_param_name = "--model", "--task", "--data_path"

    if default_model_type:
        model_param_name = "--model"
    parser.add_argument(
        model_param_name,
        type=str,
        choices=sorted(get_known_message_passing_classes()),
        default=default_model_type,
        help="GNN model type to train.",
    )
    parser.add_argument(
        task_param_name,
        type=str,
        choices=sorted(get_known_tasks()),
        default=task,
        help="Task to train model for.",
    )
    parser.add_argument(
        data_path_param_name,
        type=str,
        default=data_path,
        help="Directory containing the task data.")
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        default="zinc_vocab_1000_outputs",
        help="Path in which to store the trained model and log.",
    )
    parser.add_argument(
        "--model-params-override",
        dest="model_param_override",
        type=str,
        help="JSON dictionary overriding model hyperparameter values.",
    )
    parser.add_argument(
        "--data-params-override",
        dest="data_param_override",
        type=str,
        help="JSON dictionary overriding data hyperparameter values.",
    )
    parser.add_argument(
        "--max-epochs",
        dest="max_epochs",
        type=int,
        default=50000,
        help="Maximal number of epochs to train for.",
    )
    parser.add_argument(
        "--patience",
        dest="patience",
        type=int,
        default=25,
        help="Maximal number of epochs to continue training without improvement.",
    )
    parser.add_argument(
        "--seed",
        dest="random_seed",
        type=int,
        default=0,
        help="Random seed to use.",
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        type=str,
        help="A human-readable name for this run.",
    )
    parser.add_argument(
        "--azure-info",
        dest="azure_info",
        type=str,
        default="azure_auth.json",
        help="Azure authentication information file (JSON).",
    )
    parser.add_argument(
        "--load-saved-model",
        dest="load_saved_model",
        help="Optional location to load initial model weights from. Should be model stored in earlier run.",
    )
    parser.add_argument(
        "--load-weights-only",
        dest="load_weights_only",
        action="store_true",
        help="Optional to only load the weights of the model rather than class and dataset for further training (used in fine-tuning on pretrained network). Should be model stored in earlier run.",
    )
    parser.add_argument(
        "--disable-tf-func",
        dest="disable_tf_func",
        action="store_true",
        help="Optional to disable the building of tf function graphs and run in eager mode.",
    )
    parser.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        help="Generate less output during training.",
    )
    parser.add_argument(
        "--run-test",
        dest="run_test",
        action="store_true",
        default=False,
        help="Run on testset after training.",
    )
    parser.add_argument(
        "--azureml_logging",
        dest="azureml_logging",
        action="store_true",
        help="Log task results using AML run context.",
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines")

    parser.add_argument(
        "--hyperdrive-arg-parse",
        dest="hyperdrive_arg_parse",
        action="store_true",
        help='Enable hyperdrive argument parsing, in which unknown options "--key val" are interpreted as hyperparameter "key" with value "val".',
    )

    return parser

