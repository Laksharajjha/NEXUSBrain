"""Shared fixtures: a small world config so tests run fast."""

from __future__ import annotations

import dataclasses

import pytest

from nexus.eval.splits import Dataset, build_dataset
from nexus.world.config import WorldConfig, load_world_config


@pytest.fixture(scope="session")
def config() -> WorldConfig:
    """The default world config (full sizes)."""
    return load_world_config()


@pytest.fixture(scope="session")
def small_config(config: WorldConfig) -> WorldConfig:
    """Default world with reduced split sizes."""
    splits = dataclasses.replace(
        config.splits,
        n_train=600,
        n_dev_per_construction=30,
        n_test_novel_combination=80,
        n_test_novel_word_queries=60,
        n_test_longer=60,
        n_contradiction_probes=60,
    )
    evaluation = dataclasses.replace(config.evaluation, checkpoint_every=150)
    return dataclasses.replace(config, splits=splits, evaluation=evaluation)


@pytest.fixture(scope="session", params=[0, 1, 2])
def dataset(request: pytest.FixtureRequest, small_config: WorldConfig) -> Dataset:
    """Datasets for several seeds."""
    return build_dataset(small_config, request.param)
