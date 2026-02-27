# External model eval configs and runner for LIBERO (4 suites).
# Each model is run via its official repo; this module provides commands and optional execution.

from eval.external.configs import EXTERNAL_MODELS, get_external_config

__all__ = ["EXTERNAL_MODELS", "get_external_config"]
