"""Compare a candidate's recorded training split with an evaluation scene."""

import json
from pathlib import Path


SPLIT_KEYS = ('train_old', 'train_new', 'test_new')


def validate_training_split(model_path, current_split_path):
    """Check split image sets; return False when training metadata is unavailable."""
    model_path = Path(model_path)
    run_path = model_path / 'run.json'
    if not run_path.is_file() and (model_path.parent / 'run.json').is_file():
        # run_pipeline records metadata at output/run.json; DA checkpoints live in output/da.
        run_path = model_path.parent / 'run.json'
    if not run_path.is_file():
        print('WARNING: no run.json training split metadata; test-view separation is unverified.', flush=True)
        return False
    with run_path.open(encoding='utf-8') as f:
        run = json.load(f)
    trained = run.get('split')
    if not isinstance(trained, dict) or any(not isinstance(trained.get(k), list) for k in SPLIT_KEYS):
        print('WARNING: run.json has no usable training split metadata; test-view separation is unverified.', flush=True)
        return False
    with Path(current_split_path).open(encoding='utf-8') as f:
        current = json.load(f)
    if any(not isinstance(current.get(k), list) for k in SPLIT_KEYS):
        raise ValueError('Current split.json is missing a valid ' + ', '.join(SPLIT_KEYS) + ' view list')
    for key in SPLIT_KEYS:
        trained_names, current_names = trained[key], current[key]
        if (any(not isinstance(n, str) for n in trained_names + current_names)
                or len(trained_names) != len(set(trained_names))
                or len(current_names) != len(set(current_names))
                or set(trained_names) != set(current_names)):
            raise ValueError('Current split.json {} views differ from the training split recorded in {}'
                             .format(key, run_path))
    print('Training/evaluation split verified: old, new, and test view sets match.', flush=True)
    return True
