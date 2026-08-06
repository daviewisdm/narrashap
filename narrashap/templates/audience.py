"""Audience profiles controlling reading level and tone."""

from __future__ import annotations

AUDIENCE_PROFILES: dict[str, int] = {
    "patient": 8,
    "clinician": 14,
    "executive": 12,
}


def get_profile(audience: str) -> int:
    """Return the target Flesch-Kincaid grade level for *audience*.

    Parameters
    ----------
    audience:
        Audience key (e.g. ``"patient"``, ``"clinician"``, ``"executive"``).

    Returns
    -------
    int
        Target grade level from :data:`AUDIENCE_PROFILES`.

    Raises
    ------
    KeyError
        If *audience* is not a recognized profile key.
    """
    if audience not in AUDIENCE_PROFILES:
        valid = ", ".join(sorted(AUDIENCE_PROFILES))
        raise KeyError(
            f"Unknown audience '{audience}'. Valid audiences: {valid}"
        )
    return AUDIENCE_PROFILES[audience]
