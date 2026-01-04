"""
Serialization and persistence for FALCON-AI models.
"""

import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from ..core import FalconAI


@dataclass
class FalconCheckpoint:
    """
    Checkpoint metadata for saved FALCON models.
    """
    timestamp: str
    version: str = "0.1.0"
    metrics: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    notes: str = ""


def save_falcon(falcon: FalconAI,
                filepath: str,
                checkpoint_info: Optional[FalconCheckpoint] = None,
                include_metrics: bool = True) -> Path:
    """
    Save FALCON-AI model to disk.

    Args:
        falcon: FalconAI instance to save
        filepath: Path to save to (without extension)
        checkpoint_info: Optional checkpoint metadata
        include_metrics: Whether to include performance metrics

    Returns:
        Path to saved file
    """
    filepath = Path(filepath)

    # Gather state
    state = {
        'perception': falcon.perception,
        'decision': falcon.decision,
        'correction': falcon.correction,
        'energy_manager': falcon.energy_manager,
        'memory': falcon.memory,
        'monitor': falcon.monitor if falcon.enable_monitoring else None
    }

    # Add checkpoint info
    if checkpoint_info is None:
        checkpoint_info = FalconCheckpoint(
            timestamp=datetime.now().isoformat(),
            metrics=falcon.get_status() if include_metrics else None
        )

    state['checkpoint'] = asdict(checkpoint_info)

    # Save as pickle
    model_path = filepath.with_suffix('.falcon')
    with open(model_path, 'wb') as f:
        pickle.dump(state, f)

    # Save human-readable metadata
    metadata_path = filepath.with_suffix('.json')
    metadata = {
        'timestamp': checkpoint_info.timestamp,
        'version': checkpoint_info.version,
        'notes': checkpoint_info.notes,
        'metrics': checkpoint_info.metrics
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=_json_default)

    print(f"[OK] FALCON model saved to {model_path}")
    print(f"[OK] Metadata saved to {metadata_path}")

    return model_path


def load_falcon(filepath: str, verify_checkpoint: bool = True) -> FalconAI:
    """
    Load FALCON-AI model from disk.

    Args:
        filepath: Path to .falcon file
        verify_checkpoint: Whether to verify checkpoint metadata

    Returns:
        Loaded FalconAI instance
    """
    filepath = Path(filepath)

    if not filepath.suffix:
        filepath = filepath.with_suffix('.falcon')

    if not filepath.exists():
        raise FileNotFoundError(f"FALCON model not found: {filepath}")

    # Load state
    with open(filepath, 'rb') as f:
        state = pickle.load(f)

    # Reconstruct FALCON
    falcon = FalconAI(
        perception=state['perception'],
        decision=state['decision'],
        correction=state['correction'],
        energy_manager=state['energy_manager'],
        memory=state['memory'],
        enable_monitoring=state['monitor'] is not None
    )

    if state['monitor'] is not None:
        falcon.monitor = state['monitor']

    # Print checkpoint info
    if 'checkpoint' in state:
        checkpoint = state['checkpoint']
        print(f"[OK] Loaded FALCON from checkpoint:")
        print(f"  Timestamp: {checkpoint.get('timestamp', 'unknown')}")
        print(f"  Version: {checkpoint.get('version', 'unknown')}")
        if checkpoint.get('notes'):
            print(f"  Notes: {checkpoint['notes']}")

    return falcon


def export_metrics(falcon: FalconAI, filepath: str) -> Path:
    """
    Export FALCON metrics to JSON.

    Args:
        falcon: FalconAI instance
        filepath: Path to save metrics

    Returns:
        Path to saved file
    """
    filepath = Path(filepath).with_suffix('.json')

    metrics = falcon.get_status()
    metrics['export_timestamp'] = datetime.now().isoformat()

    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    print(f"[OK] Metrics exported to {filepath}")
    return filepath


def import_metrics(filepath: str) -> Dict[str, Any]:
    """
    Import FALCON metrics from JSON.

    Args:
        filepath: Path to metrics file

    Returns:
        Dictionary of metrics
    """
    filepath = Path(filepath).with_suffix('.json')

    with open(filepath, 'r') as f:
        metrics = json.load(f)

    return metrics


def _json_default(obj: Any):
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, set):
        return list(obj)
    return str(obj)
