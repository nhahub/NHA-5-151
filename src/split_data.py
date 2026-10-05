import csv
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import get_loaders


DATA_DIRS = (
    ROOT / "data" / "raw" / "EuroSAT_RGB" / "EuroSAT_RGB",
    ROOT / "data" / "raw" / "EuroSAT_RGB",
)
RANDOM_SEED = 42
MAX_CLASS_RATIO_DEVIATION_PERCENT = 1.0


def main():
    data_dir = next((path for path in DATA_DIRS if path.is_dir()), None)
    if data_dir is None:
        print(
            "EuroSAT RGB data was not found. Download it first with "
            "'python src/download_eurosat.py'.",
            file=sys.stderr,
        )
        return 1

    try:
        train_loader, val_loader, test_loader, class_names = get_loaders(
            data_dir=data_dir,
            seed=RANDOM_SEED,
        )
    except FileNotFoundError as exc:
        print(
            f"Could not load EuroSAT data from {data_dir}: {exc}\n"
            "Download or extract the dataset with 'python src/download_eurosat.py'.",
            file=sys.stderr,
        )
        return 1

    loaders = (
        ("Train", train_loader),
        ("Validation", val_loader),
        ("Test", test_loader),
    )
    source_targets = train_loader.dataset.base.targets
    source_counts = Counter(source_targets)
    total_samples = len(source_targets)
    expected_labels = set(range(len(class_names)))
    max_deviations = {}
    has_all_classes = {}
    indices_by_split = {}
    counts_by_split = {}

    print(f"Dataset directory: {data_dir}")
    print(f"Total classes: {len(class_names)}")
    print(f"Class names: {', '.join(class_names)}")
    print(f"Total samples: {total_samples}")
    print(f"Random seed: {RANDOM_SEED}")
    print("\nClass counts before splitting:")
    for label, class_name in enumerate(class_names):
        print(f"  {class_name:24} {source_counts[label]:6}")
    print("\nSplit summary:")
    for split_name, loader in loaders:
        sample_count = len(loader.dataset)
        indices = loader.dataset.indices
        split_counts = Counter(source_targets[index] for index in indices)
        indices_by_split[split_name] = indices
        counts_by_split[split_name] = split_counts
        has_all_classes[split_name] = expected_labels.issubset(split_counts)
        deviations = [
            abs(
                split_counts[label] / sample_count
                - source_counts[label] / total_samples
            )
            * 100
            for label in sorted(expected_labels)
        ] if sample_count else [float("inf")]
        max_deviations[split_name] = max(deviations, default=0.0)
        split_percent = sample_count / total_samples * 100
        print(
            f"  {split_name:10} {len(loader):4} batches, "
            f"{sample_count:6} samples ({split_percent:5.1f}%), "
            f"max class-ratio deviation {max_deviations[split_name]:.3f} pp"
        )

    print("\nClass counts (Train / Validation / Test):")
    for label, class_name in enumerate(class_names):
        print(
            f"  {class_name:24} "
            f"{counts_by_split['Train'][label]:6} / "
            f"{counts_by_split['Validation'][label]:6} / "
            f"{counts_by_split['Test'][label]:6}"
        )

    index_sets = {
        split_name: set(indices)
        for split_name, indices in indices_by_split.items()
    }
    no_index_overlap = all(
        index_sets[first].isdisjoint(index_sets[second])
        for first, second in (
            ("Train", "Validation"),
            ("Train", "Test"),
            ("Validation", "Test"),
        )
    )
    complete_partition = (
        set.union(*index_sets.values()) == set(range(total_samples))
        and sum(len(indices) for indices in indices_by_split.values()) == total_samples
    )
    sample_paths = [Path(path).resolve() for path, _ in train_loader.dataset.base.samples]
    path_sets = {
        split_name: {sample_paths[index] for index in indices}
        for split_name, indices in indices_by_split.items()
    }
    no_path_overlap = all(
        path_sets[first].isdisjoint(path_sets[second])
        for first, second in (
            ("Train", "Validation"),
            ("Train", "Test"),
            ("Validation", "Test"),
        )
    )

    stratification_passed = all(
        has_all_classes[split_name]
        and max_deviations[split_name] <= MAX_CLASS_RATIO_DEVIATION_PERCENT
        for split_name, _ in loaders
    )
    if stratification_passed and no_index_overlap and no_path_overlap and complete_partition:
        print(
            "\nStratification check: PASS - each split contains every class, "
            f"and each class ratio is within {MAX_CLASS_RATIO_DEVIATION_PERCENT:.1f} "
            "percentage point of the full dataset."
        )
        print("Leakage check: PASS - no image path or sample index overlaps splits.")
        print("Partition check: PASS - every source sample appears exactly once.")
    else:
        print(
            "\nValidation: FAIL - check class coverage, class ratios, overlap, "
            "and complete partition.",
            file=sys.stderr,
        )
        return 1

    manifest_dir = ROOT / "data" / "processed" / "splits" / f"seed_{RANDOM_SEED}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split_name, indices in indices_by_split.items():
        manifest_path = manifest_dir / f"{split_name.lower()}.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
            writer = csv.DictWriter(
                manifest,
                fieldnames=("image_path", "class_name", "label_id"),
            )
            writer.writeheader()
            for index in indices:
                image_path, label = train_loader.dataset.base.samples[index]
                writer.writerow(
                    {
                        "image_path": Path(image_path).resolve()
                        .relative_to(data_dir.resolve())
                        .as_posix(),
                        "class_name": class_names[label],
                        "label_id": label,
                    }
                )

    print(f"Split manifests: {manifest_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
