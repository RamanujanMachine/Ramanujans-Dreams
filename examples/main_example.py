from dreamer import System, config
from dreamer import analysis, search, extraction, post_process
from dreamer.loading import pFq
from dreamer import log, pi
import sympy as sp

# Because of pickling format we need to define these functions here
def trajectory_compute_func(d):
    """Number of search trajectories to sample for a CMF of dimension ``d``.

    :param d: CMF dimensionality.
    :return: Trajectory count (``max(10**d, 10)``).
    """
    return max(10 ** d, 10)


def trajectory_compute_func_analysis(d):
    """Number of analysis trajectories to sample for a CMF of dimension ``d``.

    :param d: CMF dimensionality.
    :return: Trajectory count (``max(10**(d-1), 10)``).
    """
    return max(10 ** (d - 1), 10)


if __name__ == '__main__':
    config.configure(
        system={
            'EXPORT_CMFS': './CMFs',                                # export CMF as objects to directory: ./CMFs
            'EXPORT_ANALYSIS_PRIORITIES': './analysis priorities',  # export shards found in analysis into: ./analysis priorities
            'EXPORT_SEARCH_RESULTS': './search results',            # export the search results into: ./search results
            'PATH_TO_SEARCHABLES': './spaces',                       # export all shard to this directory: ./spaces
            'EXPORT_ANALYSIS_PRIORITIES_FORMAT': 'json',
            'EXPORT_SEARCHABLES_FORMAT': 'json',
            'EXPORT_SEARCH_RESULTS_FORMAT': 'json'
        },
        analysis={
            # ignore shards with less than 0.1% identified trajectories as converge to the constant
            'IDENTIFY_THRESHOLD': 1e-3,
            # number of trajectories to be auto-generated in analysis
            'NUM_TRAJECTORIES_FROM_DIM': trajectory_compute_func_analysis
        },
        extraction={
            # In this case this indicates usage of pFq symmetries utilization to reduce the number of shards
            'IGNORE_DUPLICATE_SEARCHABLES': True,
            #   'auto'      -- try exact (lrs + MILP), fall back to heuristic on timeout (DEFAULT)
            #   'exact'     -- lrs + MILP only; raises on failure
            #   'heuristic' -- ray-shooting only (Best for high dimensional CMFs)
            #   'legacy'    -- brute-force lattice scan
            'STRATEGY': 'heuristic',
            # Under 'auto': exact extractor gets EXACT_TIMEOUT_SECONDS before
            # falling back; heuristic then gets HEURISTIC_TIMEOUT_SECONDS.
            # Under 'exact'/'heuristic' alone, only the matching knob applies.
            'EXACT_TIMEOUT_SECONDS': 60.0,
            'HEURISTIC_TIMEOUT_SECONDS': 200.0,
            'LOAD_SHARD_CACHE': True,
            'SAMPLING_METHOD': 'pt'
        },
        search={
            # number of trajectories to be auto-generated in search if needed by the module
            'NUM_TRAJECTORIES_FROM_DIM': trajectory_compute_func,
            'DEFAULT_USES_INV_T': False,
            'MAX_TRAJECTORY_LENGTH': 15,
            'GRAD_VARIANT': 'adam',
            'GRAD_MAX_STEPS': 50,
            'TIER2_ATTRIBUTES': (),
            'GRAD_GRAD_TOL': 1e-3,
            'SA_MAX_DEPTH': 50,
            'ANNEAL_TMIN': 2e-3,
            'SAMPLING_METHOD': 'pt'
        },
        logging={
            'GENERATE_LOGS': True
        },
        post_process={
            'TIER3_ATTRIBUTES': ()
        }
    )

    System(
        function_sources=[pFq(log(2), 2, 1, -1)],
        extractor=extraction.extractor.ShardExtractorMod,
        analyzers=[analysis.AnalyzerModV1],
        # searcher=search.SimulatedAnnealingMod, # tested
        # searcher=search.GeneticSearchModV2, # tested
        # searcher=search.GradientAscentMod, # tested
        searcher=search.SearcherModV1,
        post_processor=post_process.Tier3PostProcessModV1,
    ).run(constants=[log(2)])
