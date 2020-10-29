"""Utils."""

import sys
import time
import random
import tempfile

from pathlib import Path

import git
import json
import numpy as np
import tensorflow as tf


def _dump_git(code_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with tempfile.TemporaryDirectory() as index_dir:
        repo = git.cmd.Git(str(code_dir))
        with repo.custom_environment(GIT_INDEX_FILE=str(Path(index_dir) / 'index')):
            try:
                status = repo.status()
                (output_dir / 'git-status.txt').write_text(status)
            except git.exc.GitCommandError:
                return

            repo.add('-A')
            try:
                head = repo.get_object_header('HEAD')[0]
                (output_dir / 'git-head.txt').write_text(head.decode('ascii'))

                changes = repo.diff('HEAD')
            except ValueError:
                changes = repo.diff('--cached')

            (output_dir / 'git-diff.txt').write_text(changes)

            try:
                log = repo.log()
                (output_dir / 'git-log.txt').write_text(log)
            except git.exc.GitCommandError:
                pass


def _dump_args(output_path):
    with output_path.open('wt') as config_file:
        json.dump(sys.argv, config_file)


def setup_experiment(results_path, name, seed, save_diff=True):
    """Performs experiment setup including:
        * setting the seeds for random number generators
        * creating a directory for the results
        * saving git state including code diff
        * saving script's arguments to a file

    Make sure to add all large files to `.gitignore` when `save_diff` is set to True.

    Args:
        results_path (str): parent directory for all of the experiments
        name (str): name for the experiment
        seed (int): seed for the random number generators
        save_diff (bool): saves git diff if True

    Returns:
        str: path to the directory for the results
        float: start time.
    """
    start_time = time.time()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed + 1)
        tf.random.set_seed(seed + 2)

    results_path = Path(results_path)
    log_dir = results_path / name / str(start_time)
    log_dir.mkdir(exist_ok=True, parents=True)

    if save_diff:
        script_path = Path(sys.argv[0]).expanduser().resolve()
        _dump_git(script_path.parent, log_dir / 'git')
    _dump_args(log_dir / 'args.json')

    return log_dir, start_time
