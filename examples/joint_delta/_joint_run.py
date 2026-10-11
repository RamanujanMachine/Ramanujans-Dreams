"""Shared driver for the joint-δ run files in this directory.

Evaluates selected (start, direction) pairs of a CMF with the exact code path the
analysis stage uses per trajectory — ``TrajectoryAttributesHandler.from_cmf`` (walk
depth from ``search.DEPTH_FROM_TRAJECTORY_LEN``) + ``build_trajectory_dtos`` — and
writes the resulting flat rows to ``<out>/rows.jsonl``.  Shard extraction is skipped:
RD's hyperplane extraction does not yet finish for 10-variable Meijer-G CMFs
(symbolic determinants), see context/roadmap.md.

Joint δ is numerical evidence, not proof.
"""
import json
import os
import time

import sympy as sp
from ramanujantools import Position

from dreamer import config
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler, build_trajectory_dtos, walk_depth_for,
)


def run(cmf, cmf_name, target, cases, out_dir, walk_depth=None, extra=("joint_series", "joint_casoratian")):
    """Evaluate each case and print/write its joint-δ results.

    :param cmf: ramanujantools CMF.
    :param cmf_name: Label stored on the rows.
    :param target: ``JointConstant`` (e.g. ``joint(zeta(3), zeta(4))``).
    :param cases: ``[(label, start_tuple, direction_tuple), ...]`` in ``cmf.matrices`` axis order.
    :param out_dir: Output directory (``rows.jsonl`` is appended to).
    :param walk_depth: Override the walk depth (``None`` = the pipeline default).
    :param extra: Additional registry attributes stored on each row.
    """
    config.configure(system={'OPTIMIZATION_OBJECTIVE': 'joint_delta'},
                     search={'DEFAULT_USES_INV_T': True})   # S_N = W_N^{-T}, the reference convention
    os.makedirs(out_dir, exist_ok=True)
    axes = list(cmf.matrices.keys())
    path = os.path.join(out_dir, 'rows.jsonl')
    summary = []
    for label, start, direction in cases:
        start_p = Position(dict(zip(axes, [sp.sympify(x) for x in start])))
        dir_p = Position(dict(zip(axes, direction)))
        depth = walk_depth or walk_depth_for(cmf, dir_p)
        print(f'\n=== {label}: start={start} direction={direction} depth={depth}', flush=True)
        t = time.time()
        handler = TrajectoryAttributesHandler.from_cmf(cmf, dir_p, start_p, walk_depth=depth, walk_type=1)
        print(f'trajectory matrix built in {time.time() - t:.1f}s', flush=True)
        t = time.time()
        (dto,) = build_trajectory_dtos(
            handler, cmf_id=cmf_name, shard_id=f'{cmf_name}:{label}', cmf_name=cmf_name,
            shard_encoding_str=label, start=start_p, direction=dir_p, constants=[target],
            extra_metrics=tuple(extra),
        )
        line = dto.to_json_line()
        with open(path, 'a') as fh:
            fh.write(line + '\n')
        rec = json.loads(line)
        print(f'computed in {time.time() - t:.1f}s', flush=True)
        print(f"identified (joint forms found): {rec['identified']}")
        print(f"joint_forms: {rec.get('joint_forms')}")
        print(f"joint_delta @ N={depth}: {rec.get('joint_delta')}")
        series = rec.get('joint_series')
        if series:
            for N, d in zip(series['depths'], series['delta']):
                print(f'  N={N:5d}  joint delta={d:+.6f}')
            print(f"  interval scores: {[round(x, 6) for x in series['intervals']]}")
            print(f"  LS slope: {series['ls_slope']:+.6f}")
        print(f"casoratian: {rec.get('joint_casoratian')}")
        summary.append((label, rec.get('joint_delta')))
    print('\n=== ranking by joint_delta (numerical evidence, not proof)')
    for label, d in sorted(summary, key=lambda x: -(x[1] if x[1] is not None else float('-inf'))):
        print(f'  {label:30s} {d}')
    print(f'rows written to {path}')
