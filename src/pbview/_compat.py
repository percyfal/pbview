import warnings


class ShimWarning(DeprecationWarning):
    """Emitted by temporary compatibility shims during refactorization"""


def _warn_shim(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in a future release. "
        f"Please use {new} instead.",
        ShimWarning,
        stacklevel=3,
    )
