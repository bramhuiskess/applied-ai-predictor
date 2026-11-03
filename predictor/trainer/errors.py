"""Kept dependency-free so the CLI can catch it without importing pandas."""


class TrainingError(Exception):
    pass
