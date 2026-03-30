"""Lightweight models package initializer.

Avoid importing heavy top-level modules here to prevent import-time side-effects
when tests only need `models.components`.
"""
from .components import *