import enum
import os
import pathlib
from collections import defaultdict
from itertools import chain
from multiprocessing import Queue, Process
from queue import Empty
from typing import List, Tuple, Optional, Iterator, Union, Any, DefaultDict

import numpy as np
from rdkit import Chem
from more_itertools import chunked, ichunked

from dataset.in_memory_trace_dataset import InMemoryTraceDataset, DataFold
from models.moler_generator import MoLeRGenerator
from models.moler_vae import MoLeRVae
from utils.moler_decoding_utils import DecoderSamplingMode, MoLeRDecoderState
from utils.model_utils import load_vae_model_and_dataset
from preprocessing.data_conversion_utils import remove_non_max_frags

Pathlike = Union[str, pathlib.Path]


class MoLeRRequestType(enum.Enum):
    TERMINATE = enum.auto()
    ENCODE = enum.auto()
    DECODE = enum.auto()


def _encode_from_smiles(dataset: InMemoryTraceDataset, model: MoLeRVae, smiles_list: List[str]):
    # First, parse / load SMILES strings into the dataset:
    datapoints = []
    for smiles_str in smiles_list:
        try:
            _, trace_sample = dataset.transform_smiles_to_sample(
                smiles_str, include_generation_trace=False
            )
        except ValueError as e:
            # TODO(krmaziar): In extremely rare cases (probably only for bad MoLeR checkpoints) this
            # can happen for garbage-y molecules produced during MSO optimization (which are
            # technically correct, but cannot be kekulized). Downstream MSO then fails, as it gets
            # less embeddings that it should. Try to reproduce this; then figure out what to do.
            print(f"Warning: skipping molecule {smiles_str} due to error message \n{e}")
            continue
        datapoints.append(trace_sample)
    dataset._loaded_data[DataFold.TEST] = []
    dataset.load_data_from_list(datapoints)

    # Second: encode loaded SMILES in batches:
    result = []
    for batch_features, _ in dataset.get_tensorflow_dataset(
        data_fold=DataFold.TEST, use_worker_threads=False
    ):
        # print(f"Encoding batch of size {batch_features['num_graphs_in_batch']}")
        final_node_representations = model.compute_final_node_representations(
            batch_features, training=False
        )
    
        (graph_representation_mean, _, _,) = model.compute_latent_molecule_representations(
            final_node_representations=final_node_representations,
            num_graphs=batch_features["num_graphs_in_batch"],
            node_to_graph_map=batch_features["node_to_graph_map"],
            partial_graph_to_original_graph_map=batch_features[
                "partial_graph_to_original_graph_map"
            ],
            training=False,
        )

        gnn_output = graph_representation_mean.numpy()  # Shape [NumGraphsInBatch, LatentDim]
        result.extend(list(gnn_output))

    return result


def _decode_from_latents(
    model: Union[MoLeRVae, MoLeRGenerator],
    latent_representations: np.ndarray,
    include_latent_samples: bool = False,
    input_molecule=str,
    init_mols: List[Any] = None,
    beam_size: int = 1,
    sampling_mode: DecoderSamplingMode = DecoderSamplingMode.GREEDY,
) -> Iterator[Tuple[str, Optional[np.ndarray]]]:
    decoder_states = model.decoder.decode(
        graph_representations=latent_representations,
        input_molecule=input_molecule,
        initial_molecules=init_mols,
        beam_size=beam_size,
        sampling_mode=sampling_mode,
    )

    
    decoder_states_by_id: DefaultDict[Any, List[MoLeRDecoderState]] = defaultdict(list)
    for decoder_state in decoder_states:
        # print('111',Chem.MolToSmiles(decoder_state.molecule))
        decoder_states_by_id[decoder_state.molecule_id].append(decoder_state)

    for per_sampled_latent_results in sorted(decoder_states_by_id.items(), key=lambda kv: kv[0]):
        best_decoder_state = max(per_sampled_latent_results[1], key=lambda s: s.logprob)
        mol = remove_non_max_frags(Chem.RWMol(best_decoder_state.molecule))
        input_mol_representation = None
        if include_latent_samples:
            input_mol_representation = best_decoder_state.molecule_representation
        
        yield (Chem.MolToSmiles(mol, isomericSmiles=False), input_mol_representation)


def _moler_worker_process(
    model_path: str,
    request_queue,
    output_queue,
):
    dataset, moler_model = load_vae_model_and_dataset(model_path)

    while True:
        request, uid, arguments = request_queue.get()
        if request == MoLeRRequestType.TERMINATE or request is None:
            output_queue.put((uid, []))
            return
        elif request == MoLeRRequestType.ENCODE:
            smiles: Union[str, List[str]] = arguments
            smiles_list = smiles if isinstance(smiles, list) else [smiles]
            encoded_mols = _encode_from_smiles(dataset, moler_model, smiles_list)
            output_queue.put((uid, encoded_mols))
        elif request == MoLeRRequestType.DECODE:
            (
                latent_representations,
                include_latent_samples,
                input_molecule, 
                init_mols,
                beam_size,
                sampling_mode,
            ) = arguments
            decoder_results = _decode_from_latents(
                moler_model,
                latent_representations,
                include_latent_samples,
                input_molecule,
                init_mols,
                beam_size,
                sampling_mode,
            )
            output_queue.put((uid, list(decoder_results)))
        else:
            raise ValueError(f"Unknown worker request {request}!")


def worker_wrapper(fun, *args):
    try:
        fun(*args)
    except Exception as e:
        print(f"Worker died with {e}")
        raise


class MoLeRInferenceServer(object):
    def __init__(
        self,
        model_path: Pathlike,
        num_workers: int = 6,
        max_num_samples_per_chunk: int = 2000,
    ):
        # Make sure that our workers are only using as much memory as required:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        # Cast model_path to `str`, as it's not mutable and will be passed to different processes
        self._model_path = str(model_path)
        self._num_workers = num_workers
        self._processes: List[Process] = []
        self._max_num_samples_per_chunk = max_num_samples_per_chunk
        self.init_workers()

    def init_workers(self):
        if len(self._processes) > 0:
            return  # We are already initialised!
        self._request_queue = Queue()
        self._output_queue = Queue()
        self._processes = [
            Process(
                target=worker_wrapper,
                args=(
                    _moler_worker_process,
                    self._model_path,
                    self._request_queue,
                    self._output_queue,
                ),
            )
            for _ in range(self._num_workers)
        ]
        for worker in self._processes:
            worker.start()

    def cleanup_workers(self, ignore_failures: bool = False):
        if len(self._processes) == 0:
            return  # Nothing to clean up anymore

        issues_req_ids = set()
        for req_id, _ in enumerate(self._processes):
            # We use None as request here, as the MoLeRRequestType may already be gone in __del__:
            try:
                self._request_queue.put((None, req_id, None))
                issues_req_ids.add(req_id)
            except Exception:
                if not ignore_failures:
                    raise

        try:
            while not self._output_queue.empty():
                try:
                    execed_req_id, _ = self._output_queue.get_nowait()
                    issues_req_ids.discard(execed_req_id)
                except Empty:
                    pass
        except Exception:
            if not ignore_failures:
                raise

        # if not ignore_failures and len(issues_req_ids) > 0:
        #     raise Exception(
        #         f"Failed to clean up all worker threads! Missing worker IDs: {issues_req_ids}"
        #     )

        for worker in self._processes:
            try:
                worker.join(timeout=0.5)
                worker.close()
            except Exception:
                if not ignore_failures:
                    raise

        self._processes = []
        self._request_queue.close()
        self._output_queue.close()

    def __del__(self):
        self.cleanup_workers()

    def __enter__(self):
        self.init_workers()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.cleanup_workers(ignore_failures=True)
        return False  # Signal that exceptions should be re-raised, if needed

    def encode(self, smiles: Union[str, List[str]]):
        self.init_workers()

        smiles_list = smiles if isinstance(smiles, list) else [smiles]
        # Issue all requests to the workers, and prepare results array to hold them:
        # Choose chunk size such that all workers have something to do:
        chunk_size = min(self._max_num_samples_per_chunk, len(smiles_list) // self._num_workers + 1)
        results: List[Any] = []
        for smiles_chunk in chunked(smiles_list, chunk_size):
            self._request_queue.put((MoLeRRequestType.ENCODE, len(results), smiles_chunk))
            results.append(None)

        # Collect results and put them back into order, before returning them as one long list:
        for _ in range(len(results)):
            result_id, result = self._output_queue.get()
            results[result_id] = result

        return list(chain(*results))

    def decode(
        self,
        latent_representations: np.ndarray,
        input_molecule: str,
        include_latent_samples: bool = False,
        init_mols: List[Any] = None,
        beam_size: int = 1,
        sampling_mode: DecoderSamplingMode = DecoderSamplingMode.GREEDY,
    ) -> List[Tuple[str, Optional[np.ndarray]]]:
        self.init_workers()
        # Issue all requests to the workers, and prepare results array to hold them:
        # Choose chunk size such that all workers have something to do:
        chunk_size = min(
            self._max_num_samples_per_chunk, len(latent_representations) // self._num_workers + 1
        )
        results: List[Any] = []
        if init_mols and len(init_mols) != len(latent_representations):
            raise ValueError(
                f"Number of graph representations ({len(latent_representations)})"
                f" and initial molecules ({len(init_mols)}) needs to match!"
            )

        if not init_mols:
            init_mols = [None for _ in range(len(latent_representations))]

        init_mol_chunks = ichunked(init_mols, chunk_size)
        for latents_chunk in chunked(latent_representations, chunk_size):
            self._request_queue.put(
                (
                    MoLeRRequestType.DECODE,
                    len(results),
                    (
                        latents_chunk,
                        include_latent_samples,
                        input_molecule,
                        list(next(init_mol_chunks)),
                        beam_size,
                        sampling_mode,
                    ),
                )
            )
            results.append(None)

        # Collect results and put them back into order, before returning them as one long list:
        for _ in range(len(results)):
            result_id, result = self._output_queue.get()
            results[result_id] = result

        return list(chain(*results))
