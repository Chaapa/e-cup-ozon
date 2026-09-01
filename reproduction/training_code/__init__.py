"""Reproduction trainer for the FLV model and CatBoost selector.

The package contains no competition rows, supplementary pair surfaces, model
weights, predictions, credentials, or external-evaluation-derived data.  Every entry
point verifies caller-supplied, self-described manifests before it can fit.
"""

from .contracts import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
