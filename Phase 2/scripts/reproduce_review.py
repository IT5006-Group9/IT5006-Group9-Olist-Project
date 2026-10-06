"""Reproduce the latest review from the course ZIP in a standalone clone."""
from pathlib import Path
import argparse
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import delivery_regression as base
import feature_optimization as feature
from plot_feature_optimization import plot
from verify_feature_optimization import verify

EXPECTED_ARCHIVE_SHA256 = "90ee50730a9e799aa2d3c7b1758fef680cbbc366f61a287870144796a37156d4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Original course ZIP; no download or replacement dataset")
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve()
    if not archive.is_file():
        parser.error("Course ZIP does not exist")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != EXPECTED_ARCHIVE_SHA256:
        parser.error("ZIP differs from the pinned course version; reconcile the source before modelling")
    feature.ARCHIVE = archive
    base.prepare_data(archive=archive, output_root=ROOT / "versions/preparation_v2")
    eligible, orders, specs, cv, summary, chosen = feature.run_cv()
    comparison, predictions = feature.evaluate(eligible, orders, specs, chosen)
    plot()
    audit = verify(require_legacy_handoff=False)
    print("Selected by mature CV:", chosen)
    print(comparison.round(3).to_string(index=False))
    print("Independent checks passed:", audit["passed"], "; final test scored:", audit["test_scored"])


if __name__ == "__main__":
    main()
