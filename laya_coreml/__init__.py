"""Core ML runtime for Laya. PyTorch is only needed when converting weights."""

from .agent import Agent, load

__all__ = ["Agent", "load"]
__version__ = "0.1.0"
