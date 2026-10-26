"""Fails if anything that affects predictions changed after the configuration was frozen."""

import json

from src.extractor import DEFAULT_CONFIG, FINAL_CONFIG_NAME, CONFIGS
from src.freeze import FROZEN_PATH, fingerprint


def test_frozen_configuration_is_unchanged():
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    current = fingerprint()
    changed = sorted(k for k in frozen.keys() | current.keys() if frozen.get(k) != current.get(k))
    assert not changed, f"changed since freeze (re-run development before changing): {changed}"


def test_default_extractor_config_is_the_frozen_one():
    assert DEFAULT_CONFIG == CONFIGS[FINAL_CONFIG_NAME]
