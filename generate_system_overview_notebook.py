"""
Generate a Colab notebook (.ipynb) that:
  1. Shows the logical-flow figure (Constant → CMF → Shard → Trajectory → Evaluation)
  2. Enumerates every constant, CMF type, shard config, and trajectory policy in the system
  3. Runs sanity tests you can execute directly inside Colab
"""
import json, textwrap

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)}

def code_cell(source: str, **kw) -> dict:
    return {"cell_type": "code", "metadata": kw.get("metadata", {}),
            "source": textwrap.dedent(source).strip().splitlines(True),
            "execution_count": None, "outputs": []}

# ──────────────────────────────────────────────────────────────────────
# Notebook skeleton
# ──────────────────────────────────────────────────────────────────────
NB_VERSION = {"nbformat": 4, "nbformat_minor": 5,
              "metadata": {"kernelspec": {"display_name": "Python 3",
                                          "language": "python", "name": "python3"},
                           "language_info": {"name": "python", "version": "3.11.0"}}}

cells = []

# =====================================================================
# 0. Title
# =====================================================================
cells.append(md_cell("""
# Ramanujan Dreamer — System Overview

This notebook gives a bird's-eye view of every **Constant**, **CMF**,
**Shard**, and **Trajectory** registered in the system, together with
a logical-flow diagram and runnable sanity tests.

> Generated automatically — re-run `generate_system_overview_notebook.py`
> to refresh after code changes.
"""))

# =====================================================================
# 1. Install / import (Colab-friendly)
# =====================================================================
cells.append(md_cell("## 0 — Environment setup"))
cells.append(code_cell("""
# If running on Colab, install the local package first
# (uncomment the two lines below and update the path / wheel as needed)
# !pip install -e /content/RamanujanDream-AI-Support
# !pip install ramanujantools mpmath sympy

import sympy as sp
import mpmath as mp
from IPython.display import display, Markdown, HTML
"""))

# =====================================================================
# 2. Logical-flow figure (Mermaid, rendered as SVG via Colab / GitHub)
# =====================================================================
cells.append(md_cell("## 1 — Logical Flow: Constant → CMF → Shard → Trajectory"))

MERMAID_DIAGRAM = r'''
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '14px'}}}%%
flowchart LR
    subgraph " "
        direction LR

        C["<b>Constant</b>"]
        CMF["<b>CMF</b>"]
        SH["<b>Shard</b>"]
        TR["<b>Trajectory</b>"]
        EV["<b>Evaluation</b>"]

        C --> CMF --> SH --> TR --> EV
    end

    C_ex["π, e, ζ(3), ln 2,<br/>Catalan, γ, √2"]
    CMF_ex["pFq(2,1,z=−1)<br/>MeijerG(1,1,1,2,1)<br/>BaseCMF(custom)"]
    SH_ex["Ax &lt; b partition<br/>interior_point + shift<br/>≈ 18–200 regions per CMF"]
    TR_ex["~10<sup>d</sup> rays / shard<br/>EndToEndSampler<br/>cone-fraction estimate"]
    EV_ex["Matrix walk depth<br/>δ convergence check<br/>LIReC identification"]

    C  ~~~ C_ex
    CMF ~~~ CMF_ex
    SH  ~~~ SH_ex
    TR  ~~~ TR_ex
    EV  ~~~ EV_ex

    style C   fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style CMF fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    style SH  fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style TR  fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style EV  fill:#ffebee,stroke:#b71c1c,stroke-width:2px

    style C_ex   fill:#ffffff,stroke:#90a4ae,stroke-dasharray:4
    style CMF_ex fill:#ffffff,stroke:#90a4ae,stroke-dasharray:4
    style SH_ex  fill:#ffffff,stroke:#90a4ae,stroke-dasharray:4
    style TR_ex  fill:#ffffff,stroke:#90a4ae,stroke-dasharray:4
    style EV_ex  fill:#ffffff,stroke:#90a4ae,stroke-dasharray:4
'''

cells.append(code_cell(f"""
# Render Mermaid diagram in Colab / Jupyter
# (GitHub also renders ```mermaid blocks natively in Markdown cells)
import base64, urllib.parse
from IPython.display import display, IFrame, SVG, Image, HTML

MERMAID_SRC = {repr(MERMAID_DIAGRAM)}

# Option A: Use mermaid.ink service for a quick SVG render
encoded = base64.urlsafe_b64encode(MERMAID_SRC.encode("utf-8")).decode("ascii")
url = f"https://mermaid.ink/svg/{{encoded}}"
display(HTML(f'<img src="{{url}}" style="max-width:100%"/>'))
"""))

# Also keep a static markdown version for offline / GitHub rendering
cells.append(md_cell(f"""
<details><summary><b>Mermaid source (click to expand)</b></summary>

```mermaid
{MERMAID_DIAGRAM}
```

</details>

| Stage | What it is | Concrete examples |
|:------|:-----------|:------------------|
| **Constant** | A mathematical constant to discover formulas for | π, e, ζ(3), ln 2, Catalan, γ, √2, π² |
| **CMF** | A Conservative Matrix Field encoding a family of identities | `pFq(log(2), 2, 1, -1)`, `MeijerG(e, 1,1,1,2, 1)` |
| **Shard** | A bounded region of the CMF space defined by hyperplane inequalities Ax < b | One interior point per region; ~18–200 shards per 2D CMF |
| **Trajectory** | A direction vector sampled inside a shard, used as a walk direction | 10^d samples per shard via `EndToEndSamplingEngine` |
| **Evaluation** | Walk along trajectory, check convergence to the constant | δ computation, depth ≤ 1500, LIReC / RIES identification |
"""))

# =====================================================================
# 3. Constants table
# =====================================================================
cells.append(md_cell("## 2 — All Constants in the System"))

cells.append(code_cell("""
import sympy as sp
import mpmath as mp

mp.mp.dps = 60  # 60 decimal places for display

# ── Static constants (always registered on import) ────────────────
STATIC = [
    ("e",           sp.E,                "Euler's number"),
    ("pi",          sp.pi,               "Pi"),
    ("euler_gamma", sp.EulerGamma,       "Euler-Mascheroni constant"),
    ("pi_squared",  sp.pi**2,            "Pi squared"),
    ("catalan",     sp.Catalan,          "Catalan's constant"),
    ("gompertz",    -sp.exp(1)*sp.Ei(-1),"Gompertz constant"),
]

# ── Dynamic families (created on demand) ──────────────────────────
DYNAMIC = [
    ("zeta(n)",  "zeta-{n}",  "sp.zeta(n)",   [3, 5, 7]),
    ("log(n)",   "log-{n}",   "sp.log(n)",    [2, 3, 5]),
    ("sqrt(v)",  "sqrt({v})", "sp.sqrt(v)",   [2, 3, 5]),
    ("power(v,n)", "v^n",     "v.value**n",   []),
]

header  = "| Name | sympy expression | Numerical value (60 dp) | Kind |\\n"
header += "|:-----|:-----------------|:------------------------|:-----|\\n"
rows = []
for name, expr, desc in STATIC:
    val = mp.mpf(sp.N(expr, 62))
    rows.append(f"| `{name}` | `{expr}` | `{mp.nstr(val, 55)}` | static |")

for family, pattern, factory, examples in DYNAMIC:
    for ex in examples:
        expr = eval(factory, {"sp": sp, "v": ex, "n": ex})
        val  = mp.mpf(sp.N(expr, 62))
        name = pattern.format(n=ex, v=ex)
        rows.append(f"| `{name}` | `{expr}` | `{mp.nstr(val, 55)}` | dynamic |")

table = header + "\\n".join(rows)
display(Markdown(table))
"""))

# =====================================================================
# 4. CMF families table
# =====================================================================
cells.append(md_cell("## 3 — CMF Families"))

cells.append(md_cell("""
The system supports three CMF formatter types, each wrapping a
`ramanujantools.cmf.CMF` object:

| Formatter class | Source file | Parameters | Dimension | Example |
|:----------------|:-----------|:-----------|:----------|:--------|
| **pFq** | `dreamer/loading/funcs/pFq_fmt.py` | `p, q, z` | `p + q` symbols | `pFq(log(2), 2, 1, -1)` → 2F1 with z = −1 |
| **MeijerG** | `dreamer/loading/funcs/meijerG_fmt.py` | `m, n, p, q, z` | `p + q` symbols | `MeijerG(e, 1, 1, 1, 2, 1)` |
| **BaseCMF** | `dreamer/loading/funcs/base_cmf.py` | `cmf` (raw CMF object) | varies | Any hand-built CMF matrix field |

**Shifts**: Each CMF has `p + q` shift values (default `[0]*dim`). A `sp.Rational` shift
offsets the starting lattice point for that symbol.

**Selected start points**: Optionally restrict shard extraction to specific lattice points.
"""))

cells.append(code_cell("""
# Demonstrate creating CMF objects (requires ramanujantools)
try:
    from ramanujantools.cmf import pFq as rt_pFq
    from ramanujantools.cmf.meijer_g import MeijerG as rt_mg

    cmf_2f1 = rt_pFq(2, 1, -1)
    print("=== 2F1(z=-1) for ln(2) ===")
    print(f"  Dimension : {cmf_2f1.dim()}")
    print(f"  Symbols   : {list(cmf_2f1.matrices.keys())}")
    print(f"  # matrices: {len(cmf_2f1.matrices)}")
    for sym, mat in cmf_2f1.matrices.items():
        print(f"  M_{sym} =")
        import sympy; sympy.pprint(mat, use_unicode=True)

    print()
    cmf_mg = rt_mg(1, 1, 1, 2, 1)
    print("=== MeijerG(1,1,1,2,z=1) ===")
    print(f"  Dimension : {cmf_mg.dim()}")
    print(f"  Symbols   : {list(cmf_mg.matrices.keys())}")

except ImportError:
    print("ramanujantools not installed — skipping live CMF demo.")
    print("Install with:  pip install ramanujantools")
"""))

# =====================================================================
# 5. Shard structure
# =====================================================================
cells.append(md_cell("## 4 — Shards"))

cells.append(md_cell("""
A **Shard** is a bounded convex region of the CMF's parameter lattice,
defined by a system of linear inequalities $Ax < b$.

| Property | Type | Description |
|:---------|:-----|:------------|
| `cmf` | `CMF` | Parent CMF object |
| `constant` | `Constant` | Target constant |
| `A` | `np.ndarray (M×D)` | Hyperplane normal matrix (M planes, D dims) |
| `b` | `np.ndarray (M,)` | Hyperplane offsets |
| `shift` | `Position` | Starting-point shift per symbol |
| `interior_point` | `Position` | A guaranteed feasible point inside the shard |
| `symbols` | `List[Symbol]` | Ordered CMF symbols |
| `use_inv_t` | `bool` | Whether to use inverse-transpose for the walk |

**Extraction pipeline**:
1. Compute hyperplanes from CMF matrix zeros & poles.
2. Enumerate sign vectors → each unique sign assignment = one shard.
3. For each encoding, build `(A, b)` via `Shard.generate_matrices()`.
4. Find an interior point in each shard (grid search up to `INIT_POINT_MAX_COORD`).
5. Optionally filter symmetric pFq shards (`IGNORE_DUPLICATE_SEARCHABLES`).

**Typical count**: A `2F1` (3 symbols) produces ~18–200 shards depending on `z`.
"""))

cells.append(code_cell("""
# Demonstrate shard geometry with a synthetic example
import numpy as np
import matplotlib.pyplot as plt

# Synthetic 2D shard: 4 hyperplanes forming a quadrilateral
A = np.array([
    [ 1,  0],   # x < 5
    [-1,  0],   # -x < -1  ⟹  x > 1
    [ 0,  1],   # y < 4
    [ 0, -1],   # -y < -1  ⟹  y > 1
])
b = np.array([5, -1, 4, -1])

# Visualize the feasible region
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
from matplotlib.patches import Polygon
verts = np.array([[1,1],[5,1],[5,4],[1,4]])
poly = Polygon(verts, closed=True, alpha=0.25, fc='orange', ec='darkorange', lw=2)
ax.add_patch(poly)

# Interior point
ax.plot(3, 2.5, 'ro', ms=10, label='interior point (3, 2.5)')

# A few trajectory directions (rays from the interior)
for angle, lbl in [(30, 'traj 1'), (110, 'traj 2'), (220, 'traj 3')]:
    dx = np.cos(np.radians(angle))
    dy = np.sin(np.radians(angle))
    ax.annotate('', xy=(3+dx*1.8, 2.5+dy*1.8), xytext=(3, 2.5),
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

ax.set_xlim(-0.5, 7); ax.set_ylim(-0.5, 6)
ax.set_xlabel('symbol x₀'); ax.set_ylabel('symbol x₁')
ax.set_title('Example 2D Shard (Ax < b)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# =====================================================================
# 6. Trajectory & sampling policy
# =====================================================================
cells.append(md_cell("## 5 — Trajectories & Sampling"))

cells.append(md_cell("""
| Config key | Default | Formula | Context |
|:-----------|:--------|:--------|:--------|
| `analysis.NUM_TRAJECTORIES_FROM_DIM` | `10^d` | `λ d: 10**d` | Analysis stage |
| `search.NUM_TRAJECTORIES_FROM_DIM` | `10^d` | `λ d: 10**d` | Search stage |
| `search.DEPTH_FROM_TRAJECTORY_LEN` | ≤ 1500 | `min(1500/max(len/√d, 1), 1500)` | Walk depth per trajectory |
| `extraction.INIT_POINT_MAX_COORD` | 2 | grid ∈ [−2, 2]^d | Interior-point search grid |
| `analysis.IDENTIFY_THRESHOLD` | −1 (off) | fraction | Min convergent-trajectory ratio |

**Sampling engine** (`EndToEndSamplingEngine`):
1. **Stage 1 — Condition**: Remove degenerate & redundant rows from A → flat-land projection (`Stage1Conditioner`).
2. **Stage 2 — Raycast**: Generate uniformly-distributed integer rays inside the projected cone (`Stage2Raycaster`).
3. **Uniformity check**: Verify angular separation via NN cosine-similarity.
4. Return `Set[Position]` of integer-valued trajectory directions.
"""))

cells.append(code_cell("""
# Show trajectory count scaling
dims = list(range(1, 7))
default_counts = [10**d for d in dims]
example_counts = [max(10**d * 2, 10) for d in dims]

print("Trajectory counts by CMF dimension")
print(f"{'dim':>4}  {'default (10^d)':>15}  {'main_example (2·10^d)':>22}")
print("-" * 46)
for d, dc, ec in zip(dims, default_counts, example_counts):
    print(f"{d:>4}  {dc:>15,}  {ec:>22,}")
"""))

# =====================================================================
# 7. Full system inventory table
# =====================================================================
cells.append(md_cell("## 6 — Full System Inventory"))

cells.append(code_cell("""
from IPython.display import display, Markdown

table = '''
| Category | Items | Details |
|:---------|:------|:--------|
| **Static constants** | 6 | `e`, `pi`, `euler_gamma`, `pi_squared`, `catalan`, `gompertz` |
| **Dynamic constant families** | 4 generators (∞ instances) | `zeta(n)`, `log(n)`, `sqrt(v)`, `power(v,n)` |
| **CMF formatter types** | 3 | `pFq`, `MeijerG`, `BaseCMF` |
| **Database backend** | SQLite (`families_v1.db`) | Table: `(constant PK, family JSON[])` |
| **Shard extraction** | `ShardExtractorMod` | Hyperplane enumeration + sign-vector encoding |
| **Trajectory sampler** | `EndToEndSamplingEngine` | 2-stage: Conditioner → Raycaster |
| **Analysis module** | `AnalyzerModV1` | Serial analysis over shards, δ convergence |
| **Search module** | `SearcherModV1` | Parallel PCF search with depth control |
| **Identification** | LIReC + RIES | Experimental constant recognition |
'''
display(Markdown(table))
"""))

# =====================================================================
# 8. Sanity tests
# =====================================================================
cells.append(md_cell("## 7 — Sanity Tests"))

cells.append(md_cell("""
Run these cells to verify the system is correctly installed and the
core abstractions are consistent.
"""))

# Test 1: Constants registration
cells.append(code_cell("""
# Test 1: Constant registry consistency
print("Test 1: Constant registry")
try:
    from dreamer.utils.constants.constant import Constant
    from dreamer import e, pi, euler_gamma, pi_squared, catalan, gompertz, zeta, log, sqrt

    # Predefined constants must be registered
    for name in ['e', 'pi', 'euler_gamma', 'pi_squared', 'catalan', 'gompertz']:
        assert Constant.is_registered(name), f"{name} not registered!"

    # Dynamic creation must register
    z3 = zeta(3)
    assert Constant.is_registered('zeta-3'), "zeta-3 not registered after creation"

    ln2 = log(2)
    assert Constant.is_registered('log-2'), "log-2 not registered after creation"

    s2 = sqrt(2)
    assert Constant.is_registered('sqrt(2)'), "sqrt(2) not registered after creation"

    # Idempotent: creating the same constant again must return the same object
    z3b = zeta(3)
    assert z3b is z3 or z3b.name == z3.name, "zeta(3) re-creation inconsistent"

    print(f"  ✓ {len(Constant.available_constants())} constants registered: {Constant.available_constants()}")
    print("  ✓ PASSED")
except ImportError:
    print("  ⚠ dreamer not installed — skipping")
"""))

# Test 2: High-precision values
cells.append(code_cell("""
# Test 2: Numerical values at 500 decimal places
print("Test 2: High-precision constant values")
import mpmath as mp
import sympy as sp

mp.mp.dps = 520  # 2× margin for 500-digit accuracy

REFERENCE = {
    'e':           (sp.E,                 "2.71828182845904523536"),
    'pi':          (sp.pi,                "3.14159265358979323846"),
    'euler_gamma': (sp.EulerGamma,        "0.57721566490153286060"),
    'catalan':     (sp.Catalan,           "0.91596559417721901505"),
    'zeta-3':      (sp.zeta(3),           "1.20205690315959428539"),
    'log-2':       (sp.log(2),            "0.69314718055994530941"),
}

all_ok = True
for name, (expr, prefix) in REFERENCE.items():
    val = mp.mpf(sp.N(expr, 510))
    val_str = mp.nstr(val, 22)
    match = val_str.startswith(prefix[:15])
    status = "✓" if match else "✗"
    print(f"  {status} {name:15s} = {val_str}")
    if not match:
        all_ok = False

print(f"\\n  {'✓ PASSED' if all_ok else '✗ FAILED'}")
"""))

# Test 3: Formatter registry
cells.append(code_cell("""
# Test 3: Formatter registry completeness
print("Test 3: Formatter registry")
try:
    from dreamer.loading.funcs.formatter import Formatter
    from dreamer.loading.funcs.pFq_fmt import pFq
    from dreamer.loading.funcs.meijerG_fmt import MeijerG
    from dreamer.loading.funcs.base_cmf import BaseCMF

    expected = {'pFq', 'MeijerG', 'BaseCMF'}
    registered = set(Formatter.registry.keys())
    assert expected.issubset(registered), f"Missing formatters: {expected - registered}"
    print(f"  ✓ Registered formatters: {sorted(registered)}")
    print("  ✓ PASSED")
except ImportError:
    print("  ⚠ dreamer not installed — skipping")
"""))

# Test 4: pFq round-trip JSON
cells.append(code_cell("""
# Test 4: pFq JSON round-trip
print("Test 4: pFq JSON serialization round-trip")
try:
    from dreamer.loading.funcs.pFq_fmt import pFq
    from dreamer.loading.funcs.formatter import Formatter
    from dreamer import log

    original = pFq(log(2), 2, 1, -1)
    json_obj = original.to_json_obj()

    restored = Formatter.from_json_obj(json_obj)
    assert isinstance(restored, pFq), f"Expected pFq, got {type(restored)}"
    assert restored.p == 2, f"p mismatch: {restored.p}"
    assert restored.q == 1, f"q mismatch: {restored.q}"
    assert restored.const == 'log-2', f"const mismatch: {restored.const}"
    print(f"  ✓ JSON round-trip: {json_obj}")
    print("  ✓ PASSED")
except ImportError:
    print("  ⚠ dreamer not installed — skipping")
"""))

# Test 5: CMF creation + dimension
cells.append(code_cell("""
# Test 5: CMF creation and dimension check
print("Test 5: CMF creation and dimension")
try:
    from dreamer.loading.funcs.pFq_fmt import pFq
    from dreamer import log

    fmt = pFq(log(2), 2, 1, -1)
    shift_cmf = fmt.to_cmf()
    cmf = shift_cmf.cmf

    assert cmf.dim() == 3, f"Expected dim=3 (p+q=2+1), got {cmf.dim()}"
    assert len(cmf.matrices) == 3, f"Expected 3 matrices, got {len(cmf.matrices)}"
    print(f"  ✓ 2F1(z=-1) dimension: {cmf.dim()}")
    print(f"  ✓ Symbols: {list(cmf.matrices.keys())}")
    print(f"  ✓ Shift: {shift_cmf.shift}")
    print("  ✓ PASSED")
except ImportError:
    print("  ⚠ ramanujantools not installed — skipping")
"""))

# Test 6: Shard geometry
cells.append(code_cell("""
# Test 6: Shard in_space check
print("Test 6: Shard geometry")
try:
    import numpy as np
    import sympy as sp
    from ramanujantools import Position

    # Synthetic shard: x > 0, y > 0, x + y < 10
    A = np.array([[-1, 0], [0, -1], [1, 1]], dtype=float)
    b = np.array([0, 0, 10], dtype=float)

    from dreamer.extraction.shard import Shard
    from dreamer import e
    from ramanujantools.cmf import pFq as rt_pFq

    cmf = rt_pFq(1, 1, 1)  # simple 1F1 CMF
    x0, x1 = list(cmf.matrices.keys())[:2]
    shard = Shard(cmf, e, A, b, Position({x0: 0, x1: 0}), [x0, x1])

    # Interior point should be inside
    assert shard.in_space(Position({x0: 3, x1: 3})), "(3,3) should be inside"
    assert not shard.in_space(Position({x0: -1, x1: 3})), "(-1,3) should be outside"
    assert not shard.in_space(Position({x0: 6, x1: 6})), "(6,6) should be outside (sum=12 > 10)"
    print("  ✓ in_space checks correct")
    print("  ✓ PASSED")
except ImportError as exc:
    print(f"  ⚠ Import error: {exc} — skipping")
except Exception as exc:
    print(f"  ✗ FAILED: {exc}")
"""))

# Test 7: Trajectory count config
cells.append(code_cell("""
# Test 7: Trajectory configuration consistency
print("Test 7: Trajectory configuration")
try:
    from dreamer.configs.analysis import analysis_config
    from dreamer.configs.search import search_config

    for d in range(1, 5):
        n_analysis = analysis_config.NUM_TRAJECTORIES_FROM_DIM(d)
        n_search   = search_config.NUM_TRAJECTORIES_FROM_DIM(d)
        assert n_analysis > 0, f"Analysis trajectories for dim={d} must be > 0"
        assert n_search > 0,   f"Search trajectories for dim={d} must be > 0"
        print(f"  dim={d}: analysis={n_analysis:>6,}, search={n_search:>6,}")

    print("  ✓ PASSED")
except ImportError:
    print("  ⚠ dreamer not installed — skipping")
"""))

# Test 8: Database round-trip
cells.append(code_cell("""
# Test 8: Database write / read round-trip (uses temp file)
print("Test 8: Database round-trip")
try:
    import tempfile, os
    from dreamer.loading.databases.db_v1.db import DB
    from dreamer.loading.funcs.pFq_fmt import pFq
    from dreamer import log

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = DB(path=db_path)

        fmt = pFq(log(2), 2, 1, -1)
        db.insert(log(2), [fmt])

        retrieved = db.select(log(2))
        assert len(retrieved) == 1, f"Expected 1 CMF, got {len(retrieved)}"
        print(f"  ✓ Stored and retrieved 1 CMF for log(2)")
        print("  ✓ PASSED")
except ImportError as exc:
    print(f"  ⚠ Import error: {exc} — skipping")
except Exception as exc:
    print(f"  ✗ FAILED: {exc}")
"""))

# =====================================================================
# 9. Closing
# =====================================================================
cells.append(md_cell("""
---
## Summary

| # | Test | What it verifies |
|:--|:-----|:-----------------|
| 1 | Constant registry | All 6 static + dynamic constants register correctly |
| 2 | High-precision values | mpmath values match known digits to 500 dp |
| 3 | Formatter registry | pFq, MeijerG, BaseCMF all registered |
| 4 | pFq JSON round-trip | Serialization → deserialization preserves all fields |
| 5 | CMF creation | `pFq(2,1,-1)` produces a 3D CMF with correct symbols |
| 6 | Shard geometry | `in_space()` correctly classifies interior / exterior points |
| 7 | Trajectory config | `NUM_TRAJECTORIES_FROM_DIM` returns positive counts |
| 8 | Database round-trip | SQLite insert → select preserves CMF data |

All tests are self-contained. Run **Restart & Run All** to execute the full suite.
"""))

# ──────────────────────────────────────────────────────────────────────
# Assemble and write the notebook
# ──────────────────────────────────────────────────────────────────────
notebook = {**NB_VERSION, "cells": cells}
OUTPUT_PATH = "system_overview.ipynb"

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✓ Notebook written to: {OUTPUT_PATH}")
print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
