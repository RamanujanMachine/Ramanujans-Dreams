# Definition of Done

Every task completed by the Ramanujan Agent must meet the criteria below before it is considered **done**. These criteria exist to prevent the most common failure modes: formulas that look correct but aren't, code that works on one example but fails at scale, and results that can't be reproduced.

---

## 1. Scientific Request

A scientific request is any task that produces a mathematical result: a new formula, an identity, a proof sketch, a convergence analysis, or an investigation of a conjecture.

### Deliverables

- [ ] **LaTeX summary document** using the template in `templates/scientific_report.tex`
  - **Bottom line first**: One-paragraph executive summary stating the main result.
  - **Formal statement**: The formula, identity, or theorem stated precisely with all notation defined.
  - **Examples**: At least 2 worked numerical examples demonstrating the result.
  - **Verification**: Numerical verification to **100+ decimal places** using `mpmath`, with the verification code included.
  - **Context**: How the result relates to known CMFs, PCFs, or prior work.
  - **Open questions**: What remains unproven or unexplored.

### Verification Criteria

- [ ] Every formula computed numerically and matched against a reference constant to 100+ digits.
- [ ] Edge cases tested: $n=0$, $n=1$, boundary values.
- [ ] If a convergence rate is claimed, it is measured empirically (digits per term at depth 100, 500, 1000).
- [ ] Cross-validated against `ramanujantools` library output where applicable.
- [ ] All code used for verification is included and runnable.

### Not Done If

- Any formula is only symbolically derived but not numerically verified.
- The LaTeX document compiles with errors.
- Examples are stated but not computed.

---

## 2. Code Development

A code development task produces new or modified Python code: a function, a module, an optimization, a C extension, a script, or a bug fix.

### Deliverables

- [ ] **Working code** that passes all tests.
- [ ] **Tests** — see [Coverage Policy](COVERAGE_POLICY.md):
  - Every new public function has at least one test.
  - Tests cover: normal operation, edge cases, and at least one known-answer verification.
  - Tests are runnable via `pytest`.
  - All test functions and classes are well-documented, including assumptions and failure mode rationale per test.
  - After running tests, update `tests/TEST_AUDIT_REPORT.md` with two explicit sections:
    1. **Touched Modules (Detailed Review):** coverage status, challenge rubric scores, and regression evidence for modules changed in this task.
    2. **Non-Touched Modules (Repository-Wide Summary):** concise status/risk scan for modules not changed in this task, with follow-up actions if applicable.
- [ ] **Coverage evidence** attached:
  - `pytest tests/ -v --cov=dreamer --cov-branch --cov-report=term-missing` is run.
  - PR includes line + branch coverage for touched modules/files.
  - If changed-file coverage is below policy targets, PR includes rationale and follow-up plan.
- [ ] **Challenge quality evidence** attached for non-trivial changes:
  - PR reports the challenge rubric score from `COVERAGE_POLICY.md` (failure path, boundary, invariant, stochastic robustness, regression trap).
- [ ] **Guardrails** built into the code:
  - Input validation at public API boundaries (type checks, range checks, dimension checks).
  - Precision guards: `mpmath.mp.dps` set appropriately, no silent use of `float`.
  - Assertions for invariants that should never be violated (e.g., matrix determinant, recurrence order).
- [ ] **Sanity checks** that run as part of the function or test:
  - Known-answer test: compute a well-known constant and verify to expected precision.
  - Consistency check: forward and inverse operations compose to identity.
  - Scale check: if the function is meant to work at $N=10000$, test it at $N=10000$.
- [ ] **Documentation** for every new class or function: a description placed after the function/class definition with the following format:
  ```python
  """
  Description of what this function/class does, and why, if it does something very irregular the user might not expect.
  :param x: Description of the parameter x (if this is a function).
  :param y: Description of the parameter y (if this is a function).
  :raises SomeError: Description of the conditions under which a SomeError is raised (if applicable).
  :return: Description of the return value (if it returns something).
  """
  ```
- [ ] **Code Standards** followed: PEP 8 style, descriptive variable names, modular design.

### Performance Tasks (additional criteria)

- [ ] Benchmark before and after, with wall-clock times reported.
- [ ] Profile output included showing the hot path.
- [ ] Tested at realistic scale (not just toy inputs).

### Not Done If

- Any test fails.
- New public functions lack tests.
- Code uses Python `float` for mathematical computations.
- No guardrails on public API inputs.
- Coverage/branch report is missing for code changes.
- Challenge rubric disclosure is missing for non-trivial changes.

---

## 3. Formula Discovery

A formula discovery task searches for new polynomial continued fractions, new CMF trajectories, or new representations of constants.

### Deliverables

- [ ] **The formula** stated precisely: polynomial coefficients, recurrence, or CMF trajectory.
- [ ] **Numerical verification** to 100+ digits at depth ≥ 1000.
- [ ] **Convergence rate**: digits per term, measured empirically.
- [ ] **Irrationality measure $\delta$** if applicable (via `cmf.delta()`).
- [ ] **Classification**: which CMF family does it belong to? Is it coboundary-equivalent to a known formula?
- [ ] **LaTeX summary** with the formula, verification, and classification.
- [ ] **Reproducible code** that generates the formula and verifies it.

### Not Done If

- The formula matches fewer than 100 digits.
- No convergence rate is reported.
- The formula is not classified against known CMFs.

---

## 4. Paper Writing

A paper writing task produces or edits LaTeX content for a research paper.

### Deliverables

- [ ] LaTeX that **compiles without errors** (`pdflatex` + `biber`).
- [ ] Consistent notation per the paper's notation table.
- [ ] All claims backed by either a proof, a citation, or a numerical verification.
- [ ] Cross-references use `\cref{}`.
- [ ] New references added to `references.bib` with descriptive keys.
- [ ] Figures generated from code (reproducible), not hand-drawn.

### Not Done If

- LaTeX compilation fails.
- Any claim is made without supporting evidence.
- Notation conflicts with the paper's existing conventions.

---

## 5. Bug Fix / Investigation

### Deliverables

- [ ] **Root cause identified** and explained.
- [ ] **Fix implemented** with a regression test that would have caught the bug.
- [ ] **No other tests broken** by the fix.

### Not Done If

- The root cause is not understood (a "fix" that works by coincidence is not done).
- No regression test is added.

---

## 6. Delivery (Apply to All Task Types That Produce Code or Docs)

Every task that produces code, tests, documentation, or configuration changes
must be **delivered as a Pull Request (PR)** on the remote repository. Local
commits alone are never sufficient.

### PR Requirements

- [ ] **Branch pushed to remote.** `git push origin <branch>` must succeed.
- [ ] **PR created on GitHub** (via `gh pr create` or the web UI) targeting `main`.
- [ ] **One PR per topic.** Split unrelated changes into separate PRs:
  - Bug fixes → `fix/<description>`
  - Tests → `test/<description>`
  - Documentation → `docs/<description>`
  - Features → `feat/<description>`
- [ ] **PR title** follows conventional commits: `fix:`, `test:`, `docs:`, `feat:`.
- [ ] **PR body** contains:
  - A summary of what changed and why.
  - A list of files modified.
  - Testing instructions or confirmation that tests pass.
  - Coverage command output (or CI link) and changed-file line/branch coverage.
  - Challenge rubric score for touched non-trivial modules.
- [ ] **All CI checks pass** (once CI is configured).

### Not Done If

- Changes exist only as local commits or local branches.
- A branch is pushed but no PR is created.
- Unrelated changes are bundled in a single PR.
- The PR has no description.

---

## Universal Rules (Apply to All Task Types)

1. **Run the code.** Never declare done without executing the code and seeing the output.
2. **State the bottom line first.** The most important result goes in the first sentence of any deliverable.
3. **Include reproducibility information.** Python version, package versions, commands to run.
4. **Flag uncertainties.** If something is conjectured but not proven, say so explicitly.
5. **Update tests.** If your change could break existing tests, run the full suite and fix any failures.
6. **Deliver as a PR.** See section 6 above — local-only work is not done.
