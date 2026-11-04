"""Bind downstream reports to the exact local input used by a model run."""
import hashlib
import json
from pathlib import Path


def verify_model_input(input_path,model_directory):
    path=Path(model_directory)/'run_manifest.json'
    if not path.is_file():raise ValueError('Missing model run manifest; input provenance cannot be verified.')
    recorded=json.loads(path.read_text())
    digest=hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    if digest!=recorded.get('input_sha256'):
        raise ValueError('Input differs from the model run. Use the exact original processed-input artifact, including row order.')
    return recorded
