"""
Script to classify embeddings from all models in the outputs folder.

This script:
1. Discovers all embedding models in the outputs directory
2. For each model, loads embeddings from .npy files
3. Maps embeddings to labels from the dataset CSV
4. Runs classification pipeline for each model
"""

import logging
import json
import pathlib
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import re
import yaml
from types import SimpleNamespace
from sklearn.model_selection import StratifiedKFold

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import bacpipe modules
from bacpipe.embedding_evaluation.classification.classify import gen_loader_obj
from bacpipe.embedding_evaluation.classification.train_classifier import (
    LinearClassifier,
    train_linear_classifier,
    inference,
    KNN,
    train_knn_classifier,
)
from bacpipe.embedding_evaluation.classification.evaluate_classifier import (
    compute_task_metrics,
)


def get_model_tag_from_dirname(dir_name):
    """Extract model tag from directory name like 2026-...___google_whale-outputs."""
    match = re.match(r"^.*___(.+)-outputs$", dir_name)
    if match:
        return match.group(1)
    return None


def extract_audio_id_from_npy(filename, model_tag=None):
    """Extract audio ID from npy filename regardless of model suffix format."""
    stem = Path(filename).stem

    if model_tag:
        suffix = f"_{model_tag}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]

    match = re.match(r"^(simulated_audio_[0-9a-fA-F]+)", stem)
    if match:
        return match.group(1)

    return stem


def load_embeddings_for_model(embeddings_dir):
    """
    Load all embeddings from a model directory.
    
    Parameters
    ----------
    embeddings_dir : Path
        Path to model embeddings directory
        
    Returns
    -------
    dict
        Dictionary mapping audio_id -> embedding array
    np.ndarray
        Stacked embeddings array
    list
        List of audio IDs in same order as stacked embeddings
    """
    model_tag = get_model_tag_from_dirname(embeddings_dir.name)
    embeddings_dict = {}
    audio_ids = []
    
    npy_files = sorted(embeddings_dir.glob('*.npy'))
    logger.info(f"Found {len(npy_files)} embedding files in {embeddings_dir.name}")
    
    for npy_file in npy_files:
        audio_id = extract_audio_id_from_npy(npy_file.name, model_tag=model_tag)
        embedding = np.load(npy_file)
        if embedding.ndim != 1:
            embedding = embedding.mean(axis=0)
        embeddings_dict[audio_id] = embedding
        audio_ids.append(audio_id)
    
    # Stack embeddings in consistent order
    stacked_embeddings = np.array([embeddings_dict[aid] for aid in audio_ids])
    
    return embeddings_dict, stacked_embeddings, audio_ids


def extract_audio_id_from_csv_path(audio_path):
    """Extract audio ID from CSV path (e.g., simulated_audio_xyz.wav -> simulated_audio_xyz)"""
    return Path(audio_path).stem


def _to_event_flag(value):
    """Convert a label value to binary event flag (0/1)."""
    if pd.isna(value):
        return 0

    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1 if float(value) > 0 else 0

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "present", "positive", "event"}:
        return 1
    if text in {"0", "false", "no", "n", "absent", "negative", "none", "noise"}:
        return 0

    try:
        return 1 if float(text) > 0 else 0
    except ValueError:
        return 0


def _assign_segment_binary_labels(annotations_df, n_segments, label_column):
    """Assign one binary label (0/1) per embedding segment."""
    labels = annotations_df[label_column].apply(_to_event_flag).astype(int).tolist()

    if len(labels) == 1:
        return labels * n_segments

    if len(labels) == n_segments:
        return labels

    if {"start", "end"}.issubset(annotations_df.columns):
        ann = annotations_df.copy()
        ann["start"] = ann["start"].astype(float)
        ann["end"] = ann["end"].astype(float)
        ann["bin_label"] = ann[label_column].apply(_to_event_flag).astype(int)
        ann = ann.sort_values(["start", "end"]).reset_index(drop=True)

        t_min = ann["start"].min()
        t_max = ann["end"].max()

        if np.isfinite(t_min) and np.isfinite(t_max) and t_max > t_min:
            segment_centers = t_min + (np.arange(n_segments) + 0.5) * (t_max - t_min) / n_segments
            assigned = []
            for center in segment_centers:
                match = ann[(ann["start"] <= center) & (center < ann["end"])]
                if not match.empty:
                    assigned.append(int(match.iloc[0]["bin_label"]))
                else:
                    interval_centers = (ann["start"].to_numpy() + ann["end"].to_numpy()) / 2.0
                    closest_idx = int(np.argmin(np.abs(interval_centers - center)))
                    assigned.append(int(ann.iloc[closest_idx]["bin_label"]))
            return assigned

    if len(labels) < n_segments:
        labels = labels + [labels[-1]] * (n_segments - len(labels))
    return labels[:n_segments]


def build_segment_level_dataset(embeddings_dict, audio_ids, dataset_csv_path, label_column='event', 
                                  snr_metadata_df=None, min_snr=None):
    """
    Build a segment-level dataset with one binary label per segment.
    
    Parameters
    ----------
    embeddings_dict : dict
        Mapping of audio_id -> embedding array
    audio_ids : list
        Audio IDs in deterministic order
    dataset_csv_path : Path
        Path to dataset CSV with labels
    label_column : str
        Column name containing labels
    snr_metadata_df : pd.DataFrame, optional
        DataFrame with metadata including 'snr' column indexed or with 'audio_id' column
    min_snr : float, optional
        Minimum SNR threshold. Audio files below this will be filtered out.
        
    Returns
    -------
    np.ndarray
        Segment-level embeddings [n_segments_total, embedding_dim]
    pd.DataFrame
        DataFrame with index aligned to segment-level embeddings
    """
    df = pd.read_csv(dataset_csv_path)
    df['audio_id'] = df['audiofilename'].apply(extract_audio_id_from_csv_path)
    sort_cols = [c for c in ["audio_id", "start", "end"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    segment_embeddings = []
    class_data = []
    out_idx = 0
    skipped_low_snr = 0

    # Debug: show what we're trying to match
    if snr_metadata_df is not None and min_snr is not None:
        logger.info(f"Attempting SNR filtering with min_snr={min_snr}")
        logger.info(f"First 5 embedding audio_ids: {audio_ids[:5]}")
        if 'audio_id' in snr_metadata_df.columns:
            logger.info(f"First 5 metadata audio_ids: {snr_metadata_df['audio_id'].head(5).tolist()}")
            # Count matches
            matches = sum(1 for aid in audio_ids if aid in snr_metadata_df['audio_id'].values)
            logger.info(f"Matching audio_ids by column: {matches}/{len(audio_ids)}")
        else:
            logger.info(f"First 5 metadata index values: {snr_metadata_df.index[:5].tolist()}")
            # Count matches by index
            matches = sum(1 for aid in audio_ids if aid in snr_metadata_df.index)
            logger.info(f"Matching audio_ids by index: {matches}/{len(audio_ids)}")

    for audio_id in audio_ids:
        # Check SNR filter if provided
        if snr_metadata_df is not None and min_snr is not None:
            snr_value = None
            if 'audio_id' in snr_metadata_df.columns:
                snr_row = snr_metadata_df[snr_metadata_df['audio_id'] == audio_id]
                if not snr_row.empty:
                    snr_value = snr_row['snr'].iloc[0]
                    logger.debug(f"Audio ID {audio_id}: SNR found = {snr_value}")
                else:
                    logger.debug(f"Audio ID {audio_id} not found in SNR metadata by audio_id column")
            elif audio_id in snr_metadata_df.index:
                snr_value = snr_metadata_df.loc[audio_id, 'snr']
                logger.debug(f"Audio ID {audio_id}: SNR found in index = {snr_value}")
            else:
                logger.debug(f"Audio ID {audio_id} not in SNR metadata index")
            
            if snr_value is not None:
                if float(snr_value) < min_snr:
                    #logger.info(f"Audio ID {audio_id} filtered out: SNR={snr_value:.2f} < min_snr={min_snr}")
                    skipped_low_snr += 1
                    continue
                else:
                    logger.debug(f"Audio ID {audio_id} passed SNR filter: SNR={snr_value:.2f} >= min_snr={min_snr}")
        
        annotations = df[df['audio_id'] == audio_id]
        if annotations.empty:
            logger.warning(f"Audio ID {audio_id} not found in dataset CSV, skipping")
            continue

        emb = np.asarray(embeddings_dict[audio_id])
        if emb.ndim == 1:
            emb = emb[np.newaxis, :]
        elif emb.ndim > 2:
            emb = emb.reshape(emb.shape[0], -1)
        n_segments = emb.shape[0]
        segment_labels = _assign_segment_binary_labels(annotations, n_segments, label_column)

        for seg_idx in range(n_segments):
            segment_embeddings.append(emb[seg_idx])
            class_data.append(
                {
                    'index': out_idx,
                    'audio_id': audio_id,
                    'segment_idx': seg_idx,
                    'label': str(int(segment_labels[seg_idx])),
                    'predefined_set': 'train',
                }
            )
            out_idx += 1

    if not segment_embeddings:
        raise ValueError(
            "No segment-level embeddings could be matched to annotation labels. "
            "Check filename parsing and dataset CSV paths."
        )

    if min_snr is not None and skipped_low_snr > 0:
        logger.info(f"Skipped {skipped_low_snr} audio files with SNR < {min_snr}")

    stacked_embeddings = np.stack(segment_embeddings, axis=0)
    
    class_df = pd.DataFrame(class_data)
    
    if class_df.empty:
        raise ValueError(
            "No embedding files could be matched to rows in the dataset CSV. "
            "Check filename parsing and dataset paths."
        )

    class_df = class_df.sample(frac=1, random_state=42).reset_index(drop=True)
    class_df['index'] = class_df.index
    class_df['predefined_set'] = 'train'
    
    class_df = class_df.set_index('index')
    
    logger.info(f"Classification dataframe: {len(class_df)} samples, "
               f"{len(class_df['label'].unique())} labels")
    logger.info("  Dataset prepared with segment-level labels")
    
    return stacked_embeddings, class_df


def summarize_cv_metrics(cv_fold_results):
    """Compute mean/std summary across folds for overall metrics."""
    metric_names = set()
    for fold_result in cv_fold_results:
        metric_names.update(fold_result.get("overall", {}).keys())

    summary = {}
    for metric_name in sorted(metric_names):
        values = [
            fold_result.get("overall", {}).get(metric_name)
            for fold_result in cv_fold_results
            if fold_result.get("overall", {}).get(metric_name) is not None
        ]
        if values:
            arr = np.asarray(values, dtype=float)
            summary[metric_name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

    return summary


def run_cross_validation(
    paths,
    stacked_embeddings,
    class_df,
    classification_config,
    device,
    n_splits=5,
    random_state=42,
):
    """Run stratified cross-validation and return segment- and file-level metrics."""
    labels = class_df["label"].astype(str).to_numpy()
    min_class_count = int(class_df["label"].value_counts().min())
    effective_splits = min(n_splits, min_class_count)

    if effective_splits < 2:
        raise ValueError(
            "Cross-validation requires at least 2 samples in each class. "
            f"Minimum class count found: {min_class_count}."
        )

    if effective_splits < n_splits:
        logger.warning(
            f"Requested {n_splits} folds, but smallest class has {min_class_count} samples. "
            f"Using {effective_splits} folds instead."
        )

    skf = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )

    classifier_name = classification_config.get("name", "linear")
    kwargs = {
        "device": device,
        "learning_rate": classification_config.get("learning_rate", 0.001),
        "batch_size": classification_config.get("batch_size", 64),
        "num_epochs": classification_config.get("num_epochs", 10),
        "shuffle": classification_config.get("shuffle", False),
        "n_neighbors": classification_config.get("n_neighbors", 15),
    }

    cv_fold_results = []

    for fold_number, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels), start=1
    ):
        fold_df = class_df.copy()
        fold_df["predefined_set"] = "train"
        fold_df.loc[test_idx, "predefined_set"] = "test"

        fold_label_file = paths.class_path / f"classification_labels_fold_{fold_number}.csv"
        fold_df.to_csv(fold_label_file)

        logger.info(
            f"Fold {fold_number}/{effective_splits}: "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )

        label2index = {label: i for i, label in enumerate(fold_df.label.astype(str).unique())}
        train_gen = gen_loader_obj("train", fold_df, stacked_embeddings, label2index, **kwargs)
        test_gen = gen_loader_obj("test", fold_df, stacked_embeddings, label2index, **kwargs)

        embed_size = stacked_embeddings[0].shape[-1]

        if classifier_name == "linear":
            classifier = LinearClassifier(in_dim=embed_size, out_dim=len(label2index))
            classifier = train_linear_classifier(classifier, train_gen,loss = "cross_entropy", **kwargs)
        elif classifier_name == "knn":
            n_neighbors = kwargs.get("n_neighbors", 15)
            if len(train_gen.dataset) <= n_neighbors:
                n_neighbors = max(1, len(train_gen.dataset) - 1)
            classifier = KNN(n_neighbors=n_neighbors)
            classifier = train_knn_classifier(classifier, train_gen, **kwargs)
            kwargs["n_neighbors"] = n_neighbors
        else:
            raise ValueError(f"Unsupported classifier config: {classifier_name}")

        y_pred, y_true, probs = inference(classifier, test_gen, config=classifier_name, **kwargs)
        segment_metrics = compute_task_metrics(y_pred, y_true, probs, label2index)

        # Aggregate segment-level predictions to file-level using max pooling:
        # if any segment is predicted as 1, the file prediction is 1.
        test_rows = test_gen.dataset.dataset.reset_index(drop=True)
        file_truth = {}
        file_pred = {}
        file_prob_pos = {}

        pos_idx = label2index.get("1", 1 if len(label2index) > 1 else 0)
        for i, row in test_rows.iterrows():
            file_id = row["audio_id"]
            true_label = int(y_true[i])
            pred_label = int(y_pred[i])

            prob_vec = probs[i]
            if isinstance(prob_vec, list):
                prob_vec = np.asarray(prob_vec)
            prob_pos = float(prob_vec[pos_idx]) if np.ndim(prob_vec) > 0 else float(prob_vec)

            if file_id not in file_truth:
                file_truth[file_id] = 0
                file_pred[file_id] = 0
                file_prob_pos[file_id] = 0.0

            file_truth[file_id] = max(file_truth[file_id], true_label)
            file_pred[file_id] = max(file_pred[file_id], pred_label)
            file_prob_pos[file_id] = max(file_prob_pos[file_id], prob_pos)

        file_ids = list(file_truth.keys())
        y_true_file = [file_truth[fid] for fid in file_ids]
        y_pred_file = [file_pred[fid] for fid in file_ids]
        probs_file = [[1.0 - file_prob_pos[fid], file_prob_pos[fid]] for fid in file_ids]

        fold_file_predictions = pd.DataFrame(
            {
                "audio_id": file_ids,
                "truth": y_true_file,
                "predicted": y_pred_file,
                "predicted_prob_event": [file_prob_pos[fid] for fid in file_ids],
            }
        )
        fold_file_predictions_path = (
            paths.class_path / f"fold_{fold_number}_test_file_predictions.csv"
        )
        fold_file_predictions.to_csv(fold_file_predictions_path, index=False)

        file_metrics = compute_task_metrics(y_pred_file, y_true_file, probs_file, label2index)

        cv_fold_results.append(
            {
                "fold": fold_number,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "segment_level": {
                    "overall": segment_metrics.get("overall", {}),
                    "per_class_accuracy": segment_metrics.get("per_class_accuracy", {}),
                    "items_per_class": segment_metrics.get("items_per_class", {}),
                },
                "file_level": {
                    "overall": file_metrics.get("overall", {}),
                    "per_class_accuracy": file_metrics.get("per_class_accuracy", {}),
                    "items_per_class": file_metrics.get("items_per_class", {}),
                    "n_files": int(len(file_ids)),
                    "test_predictions_file": fold_file_predictions_path.name,
                },
            }
        )

    return cv_fold_results, effective_splits, kwargs, classifier_name


def save_cv_results(
    output_file,
    model_name,
    dataset_csv_path,
    label_column,
    embeddings_shape,
    requested_splits,
    effective_splits,
    classifier_name,
    classifier_params,
    cv_fold_results,
):
    """Save cross-validation configuration and metrics to a JSON file."""
    payload = {
        "model_name": model_name,
        "dataset_csv_path": str(dataset_csv_path),
        "label_column": label_column,
        "embeddings_shape": [int(v) for v in embeddings_shape],
        "cross_validation": {
            "requested_n_splits": int(requested_splits),
            "effective_n_splits": int(effective_splits),
            "classifier": classifier_name,
            "parameters": classifier_params,
            "prediction_pooling": "file_label = max(segment_predictions)",
        },
        "fold_results": cv_fold_results,
        "summary": {
            "segment_level": summarize_cv_metrics(
                [
                    {"overall": fold["segment_level"]["overall"]}
                    for fold in cv_fold_results
                ]
            ),
            "file_level": summarize_cv_metrics(
                [
                    {"overall": fold["file_level"]["overall"]}
                    for fold in cv_fold_results
                ]
            ),
        },
    }

    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)


def load_settings(settings_path):
    """Load settings from YAML file"""
    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)
    return settings


def get_classification_config(settings, config_name='config_1'):
    """Extract classification configuration from settings"""
    if 'class_configs' not in settings or config_name not in settings['class_configs']:
        logger.warning(f"Classification config {config_name} not found in settings")
        return {}
    
    config = settings['class_configs'][config_name]
    if not config.get('bool', False):
        logger.warning(f"Classification config {config_name} is disabled (bool=False)")
    
    return config


def classify_model_embeddings(
    embeddings_dir,
    model_name,
    dataset_csv_path,
    output_base_dir,
    classification_config,
    device='cuda',
    label_column='event',
    n_splits=5,
    random_state=42,
    snr_metadata_df=None,
    min_snr=None,
):
    """
    Run classification for a single model's embeddings.
    
    Parameters
    ----------
    embeddings_dir : Path
        Path to model embeddings directory
    model_name : str
        Name of the model
    dataset_csv_path : Path
        Path to dataset CSV with labels
    output_base_dir : Path
        Base directory for classification outputs
    classification_config : dict
        Classification configuration from settings
    device : str
        Device to use ('cuda' or 'cpu')
    label_column : str
        Column name containing labels
    snr_metadata_df : pd.DataFrame, optional
        DataFrame with metadata including SNR
    min_snr : float, optional
        Minimum SNR threshold for filtering
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Classifying embeddings for model: {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"SNR filtering: snr_metadata_df={'Present' if snr_metadata_df is not None else 'None'}, min_snr={min_snr}")
    
    # Load embeddings
    embeddings_dict, stacked_embeddings, audio_ids = load_embeddings_for_model(embeddings_dir)

    # Build segment-level embeddings and labels.
    stacked_embeddings, class_df = build_segment_level_dataset(
        embeddings_dict, audio_ids, dataset_csv_path, label_column,
        snr_metadata_df=snr_metadata_df, min_snr=min_snr
    )
    
    # Create output paths object
    output_dir = output_base_dir / model_name / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_label_file = output_dir / "classification_labels.csv"
    class_df.to_csv(generated_label_file)
    logger.info(f"Saved base classification labels to {generated_label_file}")
    
    paths = SimpleNamespace(
        class_path=output_dir,
        plot_path=output_dir,
        labels_path=output_dir,
    )
    
    logger.info(f"Embeddings shape: {stacked_embeddings.shape}")
    
    # Run classification with cross-validation
    try:
        fold_results, effective_splits, classifier_params, classifier_name = run_cross_validation(
            paths=paths,
            stacked_embeddings=stacked_embeddings,
            class_df=class_df,
            classification_config=classification_config,
            device=device,
            n_splits=n_splits,
            random_state=random_state,
        )

        cv_output_file = output_dir / f"cv_results_{classifier_name}.json"
        save_cv_results(
            output_file=cv_output_file,
            model_name=model_name,
            dataset_csv_path=dataset_csv_path,
            label_column=label_column,
            embeddings_shape=stacked_embeddings.shape,
            requested_splits=n_splits,
            effective_splits=effective_splits,
            classifier_name=classifier_name,
            classifier_params=classifier_params,
            cv_fold_results=fold_results,
        )

        logger.info(f"Saved cross-validation results to {cv_output_file}")
        logger.info(f"✓ Classification completed for {model_name}")
        return True
    except Exception as e:
        logger.error(f"✗ Classification failed for {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def discover_and_classify_models(
    embeddings_base_dir,
    dataset_csv_path,
    bacpipe_root,
    device='cuda',
    label_column='event',
    n_splits=5,
    random_state=42,
    snr_metadata_path=None,
    min_snr=None,
):
    """
    Discover all models and run classification for each.
    
    Parameters
    ----------
    embeddings_base_dir : Path
        Base directory containing all model embeddings
    dataset_csv_path : Path
        Path to dataset CSV with labels
    bacpipe_root : Path
        Root path of bacpipe project (for saving outputs)
    device : str
        Device to use ('cuda' or 'cpu')
    label_column : str
        Column name containing labels
    snr_metadata_path : str, optional
        Path to pickle file containing metadata dataframe with SNR
    min_snr : float, optional
        Minimum SNR threshold for filtering training data
    """
    embeddings_base_dir = Path(embeddings_base_dir)
    dataset_csv_path = Path(dataset_csv_path)
    output_base_dir = embeddings_base_dir.parent / "evaluations"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Load settings
    settings_path = Path(bacpipe_root) / "bacpipe" / "settings.yaml"
    settings = load_settings(settings_path)
    classification_config = get_classification_config(settings)
    
    # Load SNR metadata if provided
    snr_metadata_df = None
    if snr_metadata_path is not None:
        logger.info(f"Loading SNR metadata from {snr_metadata_path}")
        try:
            snr_metadata_df = pd.read_pickle(snr_metadata_path)
            logger.info(f"Loaded metadata with {len(snr_metadata_df)} rows")
            logger.info(f"Metadata columns: {list(snr_metadata_df.columns)}")
            logger.info(f"Metadata index name: {snr_metadata_df.index.name}")
            
            # Ensure audio_id column exists and is normalized to stem format
            if 'audio_id' not in snr_metadata_df.columns:
                if 'audio_file' in snr_metadata_df.columns:
                    snr_metadata_df['audio_id'] = snr_metadata_df['audio_file'].apply(
                        lambda x: Path(x).stem
                    )
                    logger.info("Extracted audio_id from audio_file column")
            
            # Log sample audio_ids and SNR values
            if 'audio_id' in snr_metadata_df.columns:
                logger.info(f"Sample audio_ids from metadata: {snr_metadata_df['audio_id'].head(3).tolist()}")
            if 'snr' in snr_metadata_df.columns:
                logger.info(f"SNR value range: {snr_metadata_df['snr'].min():.2f} to {snr_metadata_df['snr'].max():.2f}")
                logger.info(f"SNR null count: {snr_metadata_df['snr'].isna().sum()}")
            
            if min_snr is not None:
                logger.info(f"Will filter files with SNR < {min_snr}")
        except Exception as e:
            logger.error(f"Failed to load SNR metadata: {e}")
            import traceback
            traceback.print_exc()
            snr_metadata_df = None
    
    # Discover all model directories
    model_dirs = sorted([d for d in embeddings_base_dir.iterdir() 
                        if d.is_dir() and not d.name.startswith('.')])
    
    logger.info(f"Found {len(model_dirs)} model directories:")
    for d in model_dirs:
        logger.info(f"  - {d.name}")
    
    # Run classification for each model
    results = {}
    for model_dir in model_dirs:
        model_name = model_dir.name
        success = classify_model_embeddings(
            embeddings_dir=model_dir,
            model_name=model_name,
            dataset_csv_path=dataset_csv_path,
            output_base_dir=output_base_dir,
            classification_config=classification_config,
            device=device,
            label_column=label_column,
            n_splits=n_splits,
            random_state=random_state,
            snr_metadata_df=snr_metadata_df,
            min_snr=min_snr,
        )
        results[model_name] = success
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Classification Summary")
    logger.info(f"{'='*60}")
    succeeded = sum(1 for v in results.values() if v)
    logger.info(f"Succeded: {succeeded}/{len(results)}")
    for model, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {model}")
    
    return results


if __name__ == "__main__":
    # Configuration
    embeddings_base_dir = "/data/bramcuyx/bacpipe_results_denoised/denoised/embeddings"
    dataset_csv_path = "/data/bramcuyx/Gitlab/bacpipe/light_20260319_dataset_denoised_singleannot.csv"
    bacpipe_root = "/data/bramcuyx/Gitlab/bacpipe"
    device = "cuda"
    label_column = "label:event"  # Column name from the CSV
    n_splits = 5
    random_state = 42
    snr_metadata_path = "/mnt/fscompute_shared/simulation_dataset/datasets/light_20260319_dataset.pkl"  # Set to pickle path of metadata dataframe with SNR column
    min_snr = -20  # Set to desired minimum SNR threshold (e.g., 5.0)

    try:
        import numpy._core.numeric  # type: ignore # noqa: F401
    except ModuleNotFoundError:
        import numpy.core.numeric as _np_core_numeric

    sys.modules["numpy._core.numeric"] = _np_core_numeric

    try:
        dataset_df = pd.read_pickle(snr_metadata_path)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not read pickle at {snr_metadata_path}. "
            "This usually means the pickle was created with a different NumPy/Pandas version."
        ) from exc
    dataset_df['audio_id'] = dataset_df['audio_file'].apply(lambda x: pathlib.Path(x).stem)
    
    # Run classification
    results = discover_and_classify_models(
        embeddings_base_dir=embeddings_base_dir,
        dataset_csv_path=dataset_csv_path,
        bacpipe_root=bacpipe_root,
        device=device,
        label_column=label_column,
        n_splits=n_splits,
        random_state=random_state,
        snr_metadata_path=snr_metadata_path,
        min_snr=min_snr,
    )
