"""Canonical factories used by training and independent evaluation."""

from .build_finetune import NCHNetBaseline, build_finetune_model

build_model = build_finetune_model

__all__ = ["NCHNetBaseline", "build_model", "build_finetune_model"]
