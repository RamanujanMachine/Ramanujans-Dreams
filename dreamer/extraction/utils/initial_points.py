from typing import Callable, Optional, Tuple
from numba import njit, types
from numba.typed import Dict
import numpy as np
import multiprocessing as mp
import itertools
from dreamer.utils.logger import Logger
from functools import partial

_worker_cache = {}

MAPPING_DICT = dict[Tuple, np.ndarray]
FILTER_FUNC_DTYPE = Callable[[MAPPING_DICT], MAPPING_DICT]


# ------------------------------------------------------------
#   pFq symmetries mapping filters
# ------------------------------------------------------------

def __same_shift_indices(shift, p, q):
    """
    Build index groups of equal fractional shifts for numerator and denominator blocks.
    :param shift: Shift vector used for pFq symmetry grouping.
    :param p: Number of numerator parameters.
    :param q: Number of denominator parameters.
    :raises ValueError: If ``p + q`` does not match ``len(shift)``.
    :return: A pair ``(p_groups, q_groups)`` of index arrays grouped by equal shift fraction.
    """
    if p + q != len(shift):
        raise ValueError(f'p + q must be the dimension of the space: {len(shift)} != {p} + {q}')
    reduced = np.array([v - int(v) for v in shift])
    reduced_nom = reduced[:p]
    reduced_denom = reduced[p:]

    p_groups = []
    for unique_offset in np.unique(reduced_nom):
        indices = np.where(reduced_nom == unique_offset)[0]
        p_groups.append(indices)

    q_groups = []
    for unique_offset in np.unique(reduced_denom):
        indices = np.where(reduced_denom == unique_offset)[0]
        q_groups.append(indices)
    return p_groups, q_groups


def filter_symmetrical_cones(mapping, p, q, shift):
    """
    Remove pFq-symmetric duplicates from a shard-to-point mapping.
    :param mapping: Dictionary mapping shard signatures to representative points.
    :param p: Number of numerator dimensions.
    :param q: Number of denominator dimensions.
    :param shift: Shift vector of length ``p + q`` used to define symmetry classes.
    :raises ValueError: If ``p + q`` does not match ``len(shift)``.
    :return: Filtered mapping with one representative per symmetric cone.
    """
    if not mapping:
        return {}

    if p + q != len(shift):
        raise ValueError(f'p + q must be the dimension of the space: {len(shift)} != {p} + {q}')

    encodings = list(mapping.keys())
    points = np.array(list(mapping.values()))
    p_part = points[:, :p]
    q_part = points[:, p:]

    p_groups, q_groups = __same_shift_indices(shift, p, q)

    canonical_points = np.empty_like(points)

    for group_indices in p_groups:
        sorted_subgroup = np.sort(p_part[:, group_indices], axis=1)
        canonical_points[:, group_indices] = sorted_subgroup

    for group_indices in q_groups:
        sorted_subgroup = np.sort(q_part[:, group_indices], axis=1)
        canonical_points[:, p + group_indices] = sorted_subgroup

    _, unique_indices = np.unique(canonical_points, axis=0, return_index=True)
    return {encodings[int(i)]: points[int(i)] for i in unique_indices}


# ------------------------------------------------------------
#   Compute shard encoding and initial point pairs
# ------------------------------------------------------------
def __is_candidate_closer(candidate: np.ndarray, current: np.ndarray) -> bool:
    """
    Compare two points and decide whether ``candidate`` is preferred.
    :param candidate: Candidate point for a shard representative.
    :param current: Current selected representative point.
    :return: ``True`` when ``candidate`` has smaller squared norm, or ties and is lexicographically smaller.
    """
    candidate_norm = int(np.dot(candidate, candidate))
    current_norm = int(np.dot(current, current))

    if candidate_norm != current_norm:
        return candidate_norm < current_norm
    return tuple(candidate.tolist()) < tuple(current.tolist())


def __generate_numba_worker(M):
    """
    Creates the appropriate numba worker for initial point generation
    :param M: Length of the shard signature
    :return: Compiled worker that maps each signature to the closest sampled point.
    """
    num_chunks = (M + 63) // 64  # each chunk may use up to 64 bits for signature
    chunk_size = 512  # make sure size fits in L1 cache for fast computing
    tuple_elements = ", ".join([f"sig_chunks[{i}]" for i in range(num_chunks)])
    tuple_str = f"({tuple_elements},)" if num_chunks == 1 else f"({tuple_elements})"

    code = f"""
def dynamic_compute_block(fixed_prefix, D, S, A, b):
    M_val = A.shape[0]
    K = len(fixed_prefix)
    rem_D = D - K

    state = np.zeros(D, dtype=np.int32)
    for i in range(K):
        state[i] = fixed_prefix[i]

    offset = S // 2

    unique_mapping = Dict.empty(
        key_type=tuple_type,
        value_type=int_array_type  # Ensure we save int arrays
    )
    unique_norms = Dict.empty(
        key_type=tuple_type,
        value_type=norm_type
    )

    BLOCK_SIZE = {chunk_size}
    block = np.zeros((BLOCK_SIZE, D), dtype=np.int64) # Pure integer block!
    total_points = np.int64(S) ** np.int64(rem_D)
    points_generated = np.int64(0)

    while points_generated < total_points:
        current_batch_size = 0

        while current_batch_size < BLOCK_SIZE and points_generated < total_points:
            for j in range(D):
                block[current_batch_size, j] = state[j] - offset
            current_batch_size += 1
            points_generated += 1
            for d in range(D - 1, K - 1, -1):   # Advance the state to smallest step
                state[d] += 1
                if state[d] < S: break
                else: state[d] = 0

        for i in range(current_batch_size):
            is_on_hyperplane = False
            sig_chunks = np.zeros({num_chunks}, dtype=np.int64)

            # Compute and update chunk signature
            for j in range(M_val):
                # substitute in the linear equation
                val = b[j]
                for d in range(D):
                    val += A[j, d] * block[i, d]

                if val == 0:
                    is_on_hyperplane = True
                    break

                if val > 0:
                    # Update chunk signature - add a 1 bit
                    chunk_idx = j // 64
                    bit_idx = np.int64(j % 64)
                    sig_chunks[chunk_idx] |= (np.int64(1) << bit_idx)

            if is_on_hyperplane:
                continue

            point_norm = np.int64(0)
            for d in range(D):
                point_norm += block[i, d] * block[i, d]

            sig_tuple = {tuple_str}
            if sig_tuple not in unique_mapping:
                # First point for this shard signature.
                unique_mapping[sig_tuple] = block[i, :].copy()
                unique_norms[sig_tuple] = point_norm
            else:
                current_norm = unique_norms[sig_tuple]
                should_replace = point_norm < current_norm

                if (not should_replace) and (point_norm == current_norm):
                    # Deterministic tie-break: keep lexicographically smallest point.
                    current_point = unique_mapping[sig_tuple]
                    for d in range(D):
                        if block[i, d] < current_point[d]:
                            should_replace = True
                            break
                        if block[i, d] > current_point[d]:
                            break

                if should_replace:
                    unique_mapping[sig_tuple] = block[i, :].copy()
                    unique_norms[sig_tuple] = point_norm
    return unique_mapping
    """
    local_env = {
        'np': np,
        'Dict': Dict,
        'tuple_type': types.UniTuple(types.int64, num_chunks),
        'int_array_type': types.int64[:],  # Use strictly int64 for the saved points
        'norm_type': types.int64,
    }
    exec(code, local_env)
    return njit(local_env['dynamic_compute_block'])


def decode_signatures(unique_tuples, M):
    """
    Decode packed shard signatures into a ``{+1, -1}`` sign matrix.
    :param unique_tuples: Iterable of encoded signature tuples.
    :param M: Number of hyperplanes
    :raises ValueError: If ``M`` is negative.
    :return: The decoded signatures
    """
    if M < 0:
        raise ValueError(f'M must be non-negative, got {M}')

    N = len(unique_tuples)
    if N == 0:
        return np.empty((0, M), dtype=np.int8)

    chunks_array = np.array(list(unique_tuples), dtype=np.int64)
    if chunks_array.ndim == 1:
        chunks_array = chunks_array.reshape(-1, 1)

    # Convert to bits matrix
    bits = np.zeros((N, M), dtype=np.int8)
    for j in range(M):
        chunk_idx = j // 64
        bit_idx = np.int64(j % 64)
        bit_val = (chunks_array[:, chunk_idx] >> bit_idx) & np.int64(1)
        bits[:, j] = bit_val

    # Convert matrix to +1/-1 matrix
    return (bits * 2) - 1


def __worker_wrapper(fixed_prefix: np.ndarray, D: int, S: int, A: np.ndarray, b: np.ndarray,
                     filter_func: Optional[FILTER_FUNC_DTYPE] = None) -> MAPPING_DICT:
    """
    Compile/cache and execute the numba worker for a single fixed prefix.
    :param fixed_prefix: Dimension reduction prefix
    :param D: Hypercube total dimensions
    :param S: Hypercube side length
    :param A: Linear equations expression matrix
    :param b: Linear equations free variables vector
    :param filter_func: Elimination function for filtering shards
    :return: A mapping from shard signature to the closest sampled point in this prefix slice.
    """
    M = A.shape[0]
    if M not in _worker_cache:
        _worker_cache[M] = __generate_numba_worker(M)

    compiled_func = _worker_cache[M]
    numba_dict = compiled_func(fixed_prefix, D, S, A, b)
    local_mapping = {key_tuple: np.array(point_array) for key_tuple, point_array in numba_dict.items()}

    if filter_func:
        local_mapping = filter_func(local_mapping)
    return local_mapping


def compute_mapping(D: int, S: int, A: np.ndarray, b: np.ndarray, prefix_dims: int = 2,
                    filter_func: Optional[FILTER_FUNC_DTYPE] = None) -> MAPPING_DICT:
    """
    Compute a shard-signature mapping using nearest-to-origin representative points.
    :param D: Dimension of the hypercube
    :param S: Side length of the hypercube
    :param A: Linear equations expression matrix
    :param b: Linear equations free variables vector
    :param prefix_dims: Manually compute prefix_dims out of the D of the points
    :param filter_func: Elimination function for filtering shards
    :raises ValueError: If dimensions are invalid or matrix/vector shapes are inconsistent.
    :return: A mapping from shard signature to its closest sampled point.
    """
    if D <= 0:
        raise ValueError(f'D must be positive, got {D}')
    if S <= 0:
        raise ValueError(f'S must be positive, got {S}')
    if A.ndim != 2:
        raise ValueError(f'A must be a 2D array, got ndim={A.ndim}')
    if A.shape[1] != D:
        raise ValueError(f'A second dimension must equal D: {A.shape[1]} != {D}')
    if b.ndim != 1:
        raise ValueError(f'b must be a 1D array, got ndim={b.ndim}')
    if A.shape[0] != b.shape[0]:
        raise ValueError(f'A row count must match len(b): {A.shape[0]} != {b.shape[0]}')

    prefix_dims = min(prefix_dims, D)
    coords = range(S)
    prefixes = list(itertools.product(coords, repeat=prefix_dims))

    tasks = []
    for prefix in prefixes:
        prefix_arr = np.array(prefix, dtype=np.int32)
        tasks.append((prefix_arr, D, S, A, b))

    # Global dictionary to hold the final results
    global_mapping = {}

    num_cores = mp.cpu_count()
    Logger(f"Launching {len(tasks)} jobs across {num_cores} cores...").log()

    with mp.Pool(num_cores) as pool:
        results = pool.starmap(partial(__worker_wrapper, filter_func=filter_func), tasks)

        # Merge dictionaries from all workers
        for local_mapping in results:
            for sig, point in local_mapping.items():
                if sig not in global_mapping or __is_candidate_closer(point, global_mapping[sig]):
                    global_mapping[sig] = point

    if filter_func:
        global_mapping = filter_func(global_mapping)
    return global_mapping
