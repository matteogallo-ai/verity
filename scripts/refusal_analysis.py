"""Refusal threshold sweep, calibration (ECE) and reliability diagram from run artefacts.

Reads the dev run artefact (an ``EvalRun`` JSON with confidence scores recorded),
sweeps the refusal threshold, picks it by the rule in ``verity.eval.calibration.
choose_threshold`` and measures the score's calibration. With ``--test``, applies the
dev-chosen threshold and post-hoc calibrators fitted on dev (isotonic, Platt) to the
test run, so nothing is tuned on test.

No model is called: everything is recomputed from the artefacts.

    uv run python scripts/refusal_analysis.py --dev RUN_DEV.json [--test RUN_TEST.json] \\
        --out-json scorecards/refusal-analysis.json --out-png docs/figures/reliability.png
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from verity.eval.calibration import (
    ScoredExample,
    calibration_pairs,
    choose_threshold,
    expected_calibration_error,
    refusal_point,
    reliability_bins,
    scored_examples,
    sweep,
)
from verity.eval.run_store import load_run

# At most one refusal in ten may hit an answerable question (methodology, "Refusal").
MIN_REFUSAL_PRECISION = 0.9
THRESHOLDS = [round(i * 0.05, 2) for i in range(21)]
N_BINS = 10

Calibrator = Callable[[float], float]


def _load(path: Path) -> list[ScoredExample]:
    return scored_examples(load_run(path).per_audit)


def _fit_calibrators(pairs: Sequence[tuple[float, bool]]) -> dict[str, Calibrator]:
    """Isotonic and Platt maps from raw score to P(answerable), fitted on ``pairs``."""
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    x = np.array([s for s, _ in pairs])
    y = np.array([int(a) for _, a in pairs])
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, y)
    platt = LogisticRegression().fit(x.reshape(-1, 1), y)
    return {
        "isotonic": lambda s: float(isotonic.predict([s])[0]),
        "platt": lambda s: float(platt.predict_proba([[s]])[0, 1]),
    }


def _point_dict(point: Any) -> dict[str, Any]:
    return {
        k: getattr(point, k) for k in ("threshold", "tp", "fp", "fn", "tn", "precision", "recall")
    }


def _calibration(pairs: Sequence[tuple[float, bool]]) -> dict[str, Any]:
    return {
        "n": len(pairs),
        "ece": expected_calibration_error(pairs, n_bins=N_BINS),
        "bins": [b.__dict__ for b in reliability_bins(pairs, n_bins=N_BINS)],
    }


def analyse(dev: list[ScoredExample], test: list[ScoredExample] | None) -> dict[str, Any]:
    curve = sweep(dev, THRESHOLDS)
    chosen = choose_threshold(curve, min_precision=MIN_REFUSAL_PRECISION)
    dev_pairs = calibration_pairs(dev)
    report: dict[str, Any] = {
        "rule": f"highest refusal recall with refusal precision >= {MIN_REFUSAL_PRECISION} "
        "on dev; ties to the lowest threshold",
        "dev": {
            "n": len(dev),
            "curve": [_point_dict(p) for p in curve],
            "chosen": _point_dict(chosen),
            "calibration": _calibration(dev_pairs),
        },
    }
    if test is not None:
        test_pairs = calibration_pairs(test)
        calibrated = {
            name: _calibration([(fn(s), a) for s, a in test_pairs])
            for name, fn in _fit_calibrators(dev_pairs).items()
        }
        report["test"] = {
            "n": len(test),
            "at_chosen_threshold": _point_dict(refusal_point(test, chosen.threshold)),
            "calibration": {"raw": _calibration(test_pairs), **calibrated},
        }
    return report


def plot(report: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split = "test" if "test" in report else "dev"
    series = report[split]["calibration"]
    series = series if split == "dev" else series["raw"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    left.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="perfect")
    xs = [b["mean_score"] for b in series["bins"] if b["n"]]
    ys = [b["answerable_fraction"] for b in series["bins"] if b["n"]]
    sizes = [20 + 8 * b["n"] for b in series["bins"] if b["n"]]
    left.plot(xs, ys, color="#3b6ea8", linewidth=1.5)
    left.scatter(xs, ys, s=sizes, color="#3b6ea8", zorder=3, label="bins (size = n)")
    ece = series["ece"]
    left.set_title(f"Reliability ({split}, n={series['n']}, ECE={ece:.3f})")
    left.set_xlabel("confidence score")
    left.set_ylabel("fraction answerable")
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.legend(loc="upper left", frameon=False)

    curve = report["dev"]["curve"]
    ts = [p["threshold"] for p in curve]
    right.plot(ts, [p["precision"] for p in curve], label="refusal precision", color="#3b6ea8")
    right.plot(ts, [p["recall"] for p in curve], label="refusal recall", color="#c0504d")
    chosen = report["dev"]["chosen"]["threshold"]
    right.axvline(chosen, linestyle=":", color="#333333", label=f"chosen {chosen:.2f}")
    right.set_title(f"Threshold sweep (dev, n={report['dev']['n']})")
    right.set_xlabel("threshold")
    right.set_ylim(0, 1.05)
    right.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, metadata={"Software": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyse(_load(args.dev), _load(args.test) if args.test else None)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    plot(report, args.out_png)
    print(f"threshold {report['dev']['chosen']['threshold']} -> {args.out_json}, {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
