# Tangent-bundle neural optimal control

Reproducibility code for *Tangent-bundle augmentation for neural optimal control:
sufficient conditions and a planar positioning cost gap* (manuscript).

An ideal planar acceleration-controlled robot returns toward its initial position
with a different target velocity. Its analytic optimal path crosses itself.
A smooth autonomous position-only field cannot attain arbitrarily small cost error;
a position–velocity neural model can cross the proved cost barrier.

## Setup

Use Python 3.11 or 3.12 in an isolated environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Verify the supplied experiment

```sh
python code/planar_steering.py
python code/xy_learning.py --summarize-only
```

The first command checks exact rational proof margins and integrates the affine
optimal field. The second independently reintegrates all six saved neural models,
checks their stored paths, and rebuilds `figures/xy_learning.pdf` and `.png`.
Inspect `validation/planar_steering.json` and `validation/xy_learning.json`:
`passed` must be true. This is verification of saved models, not retraining.
Training histories in the figure use saved independently evaluated checkpoints.
Reported gaps are floating-point estimates evaluated using the exact nonnegative
cost identity; numerical consistency checks are not global optimization certificates.

## Train from scratch

```sh
python code/xy_learning.py --output runs/fresh
```

This runs both architectures with seeds 11, 22, 33. Training is substantially
more expensive than verification. The reference trajectory is not a training target:
the optimizer minimizes acceleration effort plus soft terminal penalties.
See [the recorded protocol](results/xy_learning/PROTOCOL.txt) for the original run.
Exact training outcomes can depend on platform and numerical-library versions.

## Files

- `code/planar_steering.py`: task, analytic solution, proof-margin checks.
- `code/xy_learning.py`: networks, training, independent integration and plotting.
- `results/xy_learning/`: final weights and paths, checkpoint weights, metrics and protocol.
- `CITATION.cff`: citation metadata; no publication DOI has been assigned here.

No PyTorch, older sphere experiments, manuscript build files or virtual environments
are needed for this experiment. Generated figures and local verification reports are ignored by Git.

## Interpretation and limitations

For K=50 the analytic optimum is 2725/828 and the proved position-only cost gap
is at least 5e-5. All three supplied lifted runs are below this barrier; the three
position-only runs have larger errors. Their attained errors are not certified
minima of their network family. Position runs exhausted their evaluation budgets
and approached or reached parameter bounds. These results do not establish
convergence of training, hardware performance, or superiority over time-dependent
controllers. The exclusion concerns smooth autonomous position-only fields.

## License and provenance

Released under the [MIT License](LICENSE).
ChatGPT and Codex assisted with mathematical reasoning, code, documentation and
verification. The authors remain responsible for reviewing the released artifacts.

## Citation

Use the author metadata in [CITATION.cff](CITATION.cff). The manuscript is not
assigned a publication DOI in this repository.

Documentation was reviewed with [OSA](https://github.com/aimclub/OSA) 0.3.2
using Codex; its generated suggestions were checked and selectively edited.
