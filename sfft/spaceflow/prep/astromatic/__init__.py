"""Typed, subprocess-safe wrappers for the AstroMatic tools (SExtractor): configurations are dataclasses written directly to file,
execution is subprocess.run over argument lists with checked return codes, and workspaces are context-managed."""
from ._base import AstromaticTool, ToolNotFoundError, ToolRunError, find_binary
from .sextractor import SExtractor, SExConfig, SExResult, DEFAULT_COLUMNS

__all__ = ['AstromaticTool', 'ToolNotFoundError', 'ToolRunError', 'find_binary', 'SExtractor', 'SExConfig', 'SExResult', 'DEFAULT_COLUMNS']
