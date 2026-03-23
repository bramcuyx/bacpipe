import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def compute_binary_metrics(df):
    """Compute accuracy, precision, recall, f1 for a fold dataframe."""
    y_true = pd.to_numeric(df["truth"], errors="coerce").fillna(0).astype(int)
    y_pred = pd.to_numeric(df["predicted"], errors="coerce").fillna(0).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    pred_pos = int((y_pred == 1).sum())
    true_pos = int((y_true == 1).sum())

    accuracy = float((y_pred == y_true).mean())
    precision = _safe_div(tp, pred_pos)
    recall = _safe_div(tp, true_pos)
    f1_score = _safe_div(2.0 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def extract_model_name_from_fold_path(fold_path):
    """Extract model name from .../<model_dir>/classification/fold_*.csv path."""
    model_dir_name = fold_path.parent.parent.name
    match = re.match(r"^.*___(.+)$", model_dir_name)
    if match:
        return match.group(1)
    return model_dir_name


def extract_fold_number(fold_path):
    """Extract fold number from filenames like fold_1_test_file_predictions.csv."""
    match = re.search(r"fold[_-]?(\d+)", fold_path.name)
    if match:
        return int(match.group(1))
    return None


def plot_model_snr_performance(summary_df, output_path):
    """Plot mean +/- std performance by SNR for each model."""
    metrics = [
        ("accuracy_mean", "accuracy_std", "Accuracy"),
        ("precision_mean", "precision_std", "Precision"),
        ("recall_mean", "recall_std", "Recall"),
        ("f1_score_mean", "f1_score_std", "F1 Score"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for model_name, model_df in summary_df.groupby("model"):
        model_df = model_df.sort_values("snr")
        x = model_df["snr"].to_numpy()

        for ax, (mean_col, std_col, label) in zip(axes, metrics):
            y = model_df[mean_col].to_numpy()
            yerr = model_df[std_col].to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.5,
                markersize=4,
                capsize=2,
                label=model_name,
                alpha=0.9,
            )
            ax.set_title(label)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_ylim(-0.02, 1.02)

    for ax in axes[2:]:
        ax.set_xlabel("SNR")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Score")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)

    fig.suptitle("Model Performance vs SNR (mean ± std across folds)", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# read settings
with open("bacpipe/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

dataframe_dir = settings["dataframe_dir"]

# Compatibility shim for pickles created with different NumPy internals.
try:
    import numpy._core.numeric  # type: ignore # noqa: F401
except ModuleNotFoundError:
    import numpy.core.numeric as _np_core_numeric

    sys.modules["numpy._core.numeric"] = _np_core_numeric

try:
    dataset_df = pd.read_pickle(dataframe_dir)
except ModuleNotFoundError as exc:
    raise RuntimeError(
        f"Could not read pickle at {dataframe_dir}. "
        "This usually means the pickle was created with a different NumPy/Pandas version."
    ) from exc

if "audio_file" in dataset_df.columns and "audio_id" not in dataset_df.columns:
    dataset_df["audio_id"] = dataset_df["audio_file"].apply(lambda x: pathlib.Path(x).stem)

if "snr" not in dataset_df.columns:
    raise RuntimeError("Metadata dataframe does not contain an 'snr' column")

snr_map = (
    dataset_df[["audio_id", "snr"]]
    .dropna(subset=["audio_id", "snr"])
    .drop_duplicates(subset=["audio_id"], keep="first")
)

results_dir = pathlib.Path(settings["main_results_dir"]) / "denoised" / "evaluations"
print(results_dir)

# Search recursively in all nested subfolders.
folds = sorted(results_dir.glob("**/fold*_test_file_predictions.csv"))
print(f"Found {len(folds)} fold prediction files")

if not folds:
    raise RuntimeError(f"No fold prediction files found under {results_dir}")

fold_rows = []
for fold_path in folds:
    fold_df = pd.read_csv(fold_path)
    model_name = extract_model_name_from_fold_path(fold_path)
    fold_number = extract_fold_number(fold_path)
    if "audio_id" not in fold_df.columns:
        continue

    fold_df = fold_df.merge(snr_map, on="audio_id", how="left")
    fold_df = fold_df.dropna(subset=["snr"]).copy()
    if fold_df.empty:
        continue

    for snr_value, snr_group in fold_df.groupby("snr"):
        metrics = compute_binary_metrics(snr_group)
        fold_rows.append(
            {
                "model": model_name,
                "fold": fold_number,
                "snr": float(snr_value),
                "n_samples": int(len(snr_group)),
                **metrics,
            }
        )

if not fold_rows:
    raise RuntimeError("No fold rows with valid SNR values were found")

fold_metrics_df = (
    pd.DataFrame(fold_rows)
    .sort_values(["model", "snr", "fold"])
    .reset_index(drop=True)
)

summary_df = (
    fold_metrics_df.groupby(["model", "snr"], as_index=False)
    .agg(
        n_folds=("fold", "count"),
        n_samples_total=("n_samples", "sum"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_score_mean=("f1_score", "mean"),
        f1_score_std=("f1_score", "std"),
    )
    .fillna(0.0)
    .sort_values(["model", "snr"])
    .reset_index(drop=True)
)

print("\nPer-fold metrics (first rows):")
print(fold_metrics_df.head(20).to_string(index=False))

print("\nPer-model per-SNR mean/std across folds:")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

summary_out = results_dir / "model_snr_fold_metrics_mean_std.csv"
fold_out = results_dir / "model_snr_fold_metrics_per_fold.csv"
plot_out = results_dir / "model_snr_performance_mean_std.png"
summary_df.to_csv(summary_out, index=False)
fold_metrics_df.to_csv(fold_out, index=False)
plot_model_snr_performance(summary_df, plot_out)

print(f"\nSaved per-fold metrics to: {fold_out}")
print(f"Saved model mean/std summary to: {summary_out}")
print(f"Saved plot to: {plot_out}")



