"""
Script to classify embeddings from all models in the outputs folder.

This script:
1. Discovers all embedding models in the outputs directory
2. For each model, loads embeddings from .npy files
3. Maps embeddings to labels from the dataset CSV
4. Runs classification pipeline for each model
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import re
import yaml
from types import SimpleNamespace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import bacpipe modules
from bacpipe.embedding_evaluation.classification.classify import classification_pipeline


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
        embeddings_dict[audio_id] = embedding
        audio_ids.append(audio_id)
    
    # Stack embeddings in consistent order
    stacked_embeddings = np.array([embeddings_dict[aid] for aid in audio_ids])
    
    return embeddings_dict, stacked_embeddings, audio_ids


def extract_audio_id_from_csv_path(audio_path):
    """Extract audio ID from CSV path (e.g., simulated_audio_xyz.wav -> simulated_audio_xyz)"""
    return Path(audio_path).stem


def create_classification_dataframe(audio_ids, dataset_csv_path, label_column='event'):
    """
    Create a classification dataframe mapping embeddings to labels.
    
    Parameters
    ----------
    audio_ids : list
        List of audio IDs from embeddings
    dataset_csv_path : Path
        Path to dataset CSV with labels
    label_column : str
        Column name containing labels
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [index, audio_id, label, predefined_set]
    """
    # Load dataset CSV
    df = pd.read_csv(dataset_csv_path)
    
    # Extract audio IDs from CSV paths
    df['audio_id'] = df['audiofilename'].apply(extract_audio_id_from_csv_path)
    
    # Create classification dataframe
    class_data = []
    for idx, audio_id in enumerate(audio_ids):
        # Find matching row in dataset CSV
        match = df[df['audio_id'] == audio_id]
        
        if len(match) > 0:
            label = 0
            for row in match.itertuples():
                if getattr(row, label_column) != label:
                    label = 1
                    break

            class_data.append({
                'index': idx,
                'audio_id': audio_id,
                'label': str(label),
                'predefined_set': 'train',  # Default to train; could be split here if needed
            })
        else:
            logger.warning(f"Audio ID {audio_id} not found in dataset CSV, skipping")
    
    class_df = pd.DataFrame(class_data)
    
    if class_df.empty:
        raise ValueError(
            "No embedding files could be matched to rows in the dataset CSV. "
            "Check filename parsing and dataset paths."
        )

    # Create a simple train/test split (80/20)
    class_df = class_df.sample(frac=1, random_state=42)
    split_idx = int(len(class_df) * 0.8)
    if split_idx == 0 and len(class_df) > 1:
        split_idx = 1
    if split_idx >= len(class_df) and len(class_df) > 1:
        split_idx = len(class_df) - 1

    class_df.loc[class_df.index[:split_idx], 'predefined_set'] = 'train'
    class_df.loc[class_df.index[split_idx:], 'predefined_set'] = 'test'
    
    # Set index to match embedding array indices
    class_df = class_df.set_index('index')
    
    logger.info(f"Classification dataframe: {len(class_df)} samples, "
               f"{len(class_df['label'].unique())} labels")
    logger.info(f"  Train set: {len(class_df[class_df['predefined_set'] == 'train'])} samples")
    logger.info(f"  Test set: {len(class_df[class_df['predefined_set'] == 'test'])} samples")
    
    return class_df


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
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Classifying embeddings for model: {model_name}")
    logger.info(f"{'='*60}")
    
    # Load embeddings
    embeddings_dict, stacked_embeddings, audio_ids = load_embeddings_for_model(embeddings_dir)
    
    # Create classification dataframe
    class_df = create_classification_dataframe(
        audio_ids, dataset_csv_path, label_column
    )
    
    # Create output paths object
    output_dir = output_base_dir / model_name / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_label_file = output_dir / "classification_labels.csv"
    class_df.to_csv(generated_label_file)
    
    paths = SimpleNamespace(
        class_path=output_dir,
        plot_path=output_dir,
        labels_path=output_dir,
    )
    
    # Prepare embeddings for classification (shape should be [n_samples, embedding_dim])
    if stacked_embeddings.ndim == 3:
        # If embeddings are [n_samples, n_segments, embedding_dim], average across segments
        stacked_embeddings = np.mean(stacked_embeddings, axis=1)
    
    logger.info(f"Embeddings shape: {stacked_embeddings.shape}")
    
    # Extract classification parameters
    kwargs = {
        'device': device,
        'learning_rate': classification_config.get('learning_rate', 0.001),
        'batch_size': classification_config.get('batch_size', 64),
        'num_epochs': classification_config.get('num_epochs', 10),
        'shuffle': classification_config.get('shuffle', False),
    }
    
    # Run classification
    try:
        classification_config_name = classification_config.get('name', 'linear')
        classification_pipeline(
            paths=paths,
            embeds=stacked_embeddings,
            name=classification_config_name,
            dataset_csv_path=generated_label_file.name,
            overwrite=True,
            **kwargs
        )
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
    """
    embeddings_base_dir = Path(embeddings_base_dir)
    dataset_csv_path = Path(dataset_csv_path)
    output_base_dir = embeddings_base_dir.parent / "evaluations"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Load settings
    settings_path = Path(bacpipe_root) / "bacpipe" / "settings.yaml"
    settings = load_settings(settings_path)
    classification_config = get_classification_config(settings)
    
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
    embeddings_base_dir = "/data/bramcuyx/bacpipe_results/outputs/embeddings"
    dataset_csv_path = "/data/bramcuyx/Gitlab/bacpipe/light_20260319_dataset.csv"
    bacpipe_root = "/data/bramcuyx/Gitlab/bacpipe"
    device = "cuda"
    label_column = "label:event"  # Column name from the CSV
    
    # Run classification
    results = discover_and_classify_models(
        embeddings_base_dir=embeddings_base_dir,
        dataset_csv_path=dataset_csv_path,
        bacpipe_root=bacpipe_root,
        device=device,
        label_column=label_column,
    )
