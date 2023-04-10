"""Script to run preprocessing on lists of SMILES which already has train-test-valid split."""

import sys
sys.path.append("..")
import argparse
import enum
import logging
import os
import pathlib
from typing import Dict, List, Optional, Tuple, TypeVar, Union

from dpu_utils.utils import RichPath
import chem.atom_feature_utils as atom_utils
import chem.motif_utils as motif_utils
from preprocessing import generation_order as gen_order
from preprocessing.preprocess import preprocess_jsonl_files
from chem.molecule_dataset_utils import featurise_smiles_datapoints
from utils.cli_utils import (
    add_debug_flag,
    setup_logging,
    supress_tensorflow_warnings,
)
from utils.preprocessing_utils import save_data
os.environ["CUDA_VISIBLE_DEVICES"]="0"
logger = logging.getLogger(__name__)


# TODO(kmaziarz): Each data point is dictionary of the format {"SMILES": "<molecule>"}, as we used
# to use `csv.DictReader`. Consider refactoring this into a well-defined object.
DataEntry = Dict[str, str]
SmilesDataSet = List[DataEntry]
Pathlike = Union[str, pathlib.Path]
T = TypeVar("T")

GENERATION_ORDER_CLS = {
    "bfs": gen_order.BFSOrder,
    "bfs-random": gen_order.BFSOrderRandom,
    "canonical": gen_order.CanonicalOrder,
    "random": gen_order.RandomOrder,
    "loop-closing": gen_order.LoopClosingOrder,
}


class DatasetPath(enum.Enum):
    """The directory with a data set should contain the following files."""

    TRAIN = "train.smiles"
    VALID = "valid.smiles"
    TEST = "test.smiles"


def _read_data(path: Pathlike) -> SmilesDataSet:
    with open(path) as f:
        data = f.readlines()

    data = [{"SMILES": x.strip()} for x in data]
    return data


def _get_first_samples(lst: List[T], ratio: float) -> List[T]:
    """Returns first X% of the entries in `lst`, where X% is given by the `ratio` (should be between 0 and 1)."""
    k = int(ratio * len(lst))
    return lst[:k]


def load_smiles_data(
    data_path: Pathlike,
    n_datapoints: Optional[int] = None,
) -> Tuple[SmilesDataSet, SmilesDataSet, SmilesDataSet]:
    """Loads data sets from a given directory.

    Args:
        data_path: a directory with the train, validation and test SMILES files (structure described in DatasetPath)
        n_datapoints: the number of molecules to be used from the training data set. The validation and test
            data sets are rescaled accordingly. If None, all samples from each data set are taken

    Returns:
        training data set
        validation data set
        test data set
    """
    # Load the training data set
    data_path = pathlib.Path(data_path)
    train_path = data_path / DatasetPath.TRAIN.value
    train_data: SmilesDataSet = _read_data(train_path)

    # Log the information about the training data set
    num_train = len(train_data)
    logger.info(f"Read {num_train} datapoints from {train_path}")

    # Read validation and test data sets
    valid_data: SmilesDataSet = _read_data(data_path / DatasetPath.VALID.value)
    test_data: SmilesDataSet = _read_data(data_path / DatasetPath.TEST.value)

    # If num_datapoints is specified, we truncate all data sets proportionally
    if n_datapoints is not None:
        if n_datapoints <= 0:
            raise ValueError("num_datapoints must be positive.")
        # Get exactly num_datapoints for the training set
        train_data = train_data[:n_datapoints]
        ratio = n_datapoints / num_train

        # Truncate the validation and test data sets proportionally
        valid_data = _get_first_samples(valid_data, ratio)
        test_data = _get_first_samples(test_data, ratio)

    if not test_data or not valid_data:
        logger.warning("Test or valid set contains no points; increase num_datapoints")
        raise ValueError("Test or valid data sets are empty; increase num_datapoints")

    return train_data, valid_data, test_data


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess a dataset of SMILES strings.")
    parser.add_argument(
        "--INPUT_DIR",
        type=str,
        default='../zinc_input_dir',
        help="Directory which contains all the raw data components, including any labels.",
    )
    parser.add_argument(
        "--OUTPUT_DIR",
        type=str,
        default='../zinc_output_dir',
        help="Directory which will hold the resulting preprocessed data."
    )
    parser.add_argument(
        "--TRACE_DIR",
        type=str,
        default='../zinc_trace_dir',
        help="Directory in which we want to save the trace datasets."
    )
    parser.add_argument(
        "--num-datapoints",
        dest="num_datapoints",
        type=int,
        help="Use only the specified number of datapoints.",
    )
    parser.add_argument(
        "--motif-min-frequency",
        dest="motif_min_frequency",
        type=int,
        help="Minimum frequency for a motif to be included in the vocabulary.",
    )
    parser.add_argument(
        "--motif-max-vocab-size",
        dest="motif_max_vocab_size",
        type=int,
        default=1000,
        help="Maximum number of motifs in the vocabulary.",
    )
    parser.add_argument(
        "--motif-min-num-atoms",
        dest="motif_min_num_atoms",
        type=int,
        default=1,
        help="Minimum number of atoms for a motif to be included in the vocabulary.",
    )
    parser.add_argument(
        "--motif-keep-leaf-edges",
        dest="motif_keep_leaf_edges",
        action="store_true",
        default=False,
        help="Whether to keep leaf edges when extracting motifs.",
    )
    parser.add_argument(
        "--seed", dest="random_seed", type=int, default=0, help="Random seed to use."
    )

    default_num_processes = os.cpu_count()
    # On *nix, try to also use amount of host memory, as each worker roughly needs 6GB:
    try:
        mem_size = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        default_num_processes = min(default_num_processes, int(mem_size / (6 * 1024**3)))
    except Exception:
        pass  # This may happen on non-Unices; ignore
    parser.add_argument(
        "--num-processes",
        dest="num_processes",
        type=int,
        default=default_num_processes,
        help="Set number of parallel preprocessing worker processes.",
    )
    parser.add_argument(
        "--for-cgvae",
        dest="for_cgvae",
        action="store_true",
        help="Process data into format suitable for a CGVAE model.",
    )
    parser.add_argument(
        "--generation-order",
        dest="generation_order",
        default="bfs-random",
        choices=GENERATION_ORDER_CLS.keys(),
        help="Strategy to use when constructing the generation trace.",
    )
    parser.add_argument("--quiet", action="store_true", help="Turn off progress bar.")
    add_debug_flag(parser)

    return parser


def run_smiles_preprocessing(
    *,
    input_dir: Pathlike,
    trace_dir: Pathlike,
    output_dir: Pathlike,
    generation_order: str = 'bfs',
    motif_extraction_settings: Optional[motif_utils.MotifExtractionSettings] = None,
    num_datapoints: Optional[int] = None,
    num_processes: int = 5,
    quiet: bool = False,
    for_cgvae: bool = False,
) -> None:
    """Runs smiles processing, saving the final outputs to `trace_dir`.

    Args:
        input_dir: Input directory containing the SMILES data sets.
        trace_dir: Output directory where the processed traces are stored.
        output_dir: Output directory where the intermediate features are stored.
        generation_order: Generation order to use (see `GENERATION_ORDER_CLS`).
        motif_extraction_settings: Settings for motif extraction.
        num_datapoints: If specified, only this number of examples is used for training
            (validation and test folds are appropriately rescaled).
        num_processes: Number of worker processes to use.
        quiet: Whether to skip printing a progress bar.
        for_cgvae: Whether to use a CGVAE-compatible preprocessing scheme (as opposed to
            the default MoLeR-compatible scheme).
    """
    output_dir = str(output_dir)
    trace_dir = str(trace_dir)


    train_datapoints, valid_datapoints, test_datapoints = load_smiles_data(
        input_dir, n_datapoints=num_datapoints
    )

    logger.info(
        f"{len(train_datapoints)} train datapoints"
        f", {len(valid_datapoints)} validation datapoints"
        f", {len(test_datapoints)} test datapoints loaded"
        ", beginning featurization."
    )

    logger.info(f"Featurising data...")
    featurised_data = featurise_smiles_datapoints(
        train_data=train_datapoints,
        valid_data=valid_datapoints,
        test_data=test_datapoints,
        atom_feature_extractors=atom_utils.get_default_atom_featurisers(),
        num_processes=num_processes,
        motif_extraction_settings=motif_extraction_settings,
        quiet=quiet,
    )
    logger.info(f"Completed initializing feature extractors; featurising and saving data now.")

    save_data(
        featurised_data,
        output_dir=output_dir,
        quiet=quiet,
    )

    # TODO(kmaziarz): Maybe this should be done at an earlier point.
    generation_order_cls = GENERATION_ORDER_CLS[generation_order]

    # Now, convert data to traces.
    jsonl_directory = RichPath.create(output_dir)
    trace_directory = RichPath.create(trace_dir)

    preprocess_jsonl_files(
        jsonl_directory=jsonl_directory,
        output_directory=trace_directory,
        tie_fwd_bkwd_edges=True,
        num_processes=num_processes,
        generation_order_cls=generation_order_cls,
        MoLeR_style_trace=not for_cgvae,
        quiet=quiet,
    )


def _args_to_motif_settings(
    args: argparse.Namespace,
) -> Optional[motif_utils.MotifExtractionSettings]:
    """Construct MotifExtractionSettings if appropriate arguments are provided."""
    if args.motif_min_frequency is not None or args.motif_max_vocab_size > 0:
        motif_extraction_settings = motif_utils.MotifExtractionSettings(
            min_frequency=args.motif_min_frequency,
            min_num_atoms=args.motif_min_num_atoms,
            cut_leaf_edges=not args.motif_keep_leaf_edges,
            max_vocab_size=args.motif_max_vocab_size,
        )
    else:
        motif_extraction_settings = None
    return motif_extraction_settings


def run_from_args(args: argparse.Namespace) -> None:
    print('args', args)
    run_smiles_preprocessing(
        input_dir=args.INPUT_DIR,
        output_dir=args.OUTPUT_DIR,
        trace_dir=args.TRACE_DIR,
        generation_order=args.generation_order,
        motif_extraction_settings=_args_to_motif_settings(args),
        num_datapoints=args.num_datapoints,
        num_processes=args.num_processes,
        quiet=args.quiet,
        for_cgvae=args.for_cgvae,
    )


def main() -> None:
    supress_tensorflow_warnings()
    setup_logging()

    run_from_args(get_argparser().parse_args())


if __name__ == "__main__":
    main()
