# Fixed camera-ready models

This directory contains the 60 fitted neural-network checkpoints used for the
camera-ready real-data experiment: 40 checkpoints for the male and female BMI
tasks and 20 for California Housing.

Run `python -m real_data.bootstrap` from the repository's `python/` directory.
The runner verifies every file against `manifest.json` before deserializing it,
recreates the paper's seed-42 test split, and performs paired-bootstrap inference
without fitting a model.

These files are PyTorch pickle checkpoints. Load only checkpoints obtained from
this repository and leave hash verification enabled.
