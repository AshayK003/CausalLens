from __future__ import annotations


def format_effect(value: float) -> str:
    """Format effect size with sign (e.g. +1.23, -0.45)."""
    return f"{value:+.2f}"


def format_effect_pct(value: float) -> str:
    """Format effect percentage with sign (e.g. +12.3%).

    Note: Callers should pass absolute values if they want unsigned display.
    This function always adds a +/- sign.
    """
    return f"{value:+.1f}%"


def format_p_value(value: float, *, decimals: int = 4) -> str:
    """Format p-value to given decimal places."""
    return f"{value:.{decimals}f}"


def format_ci(lower: float, upper: float) -> str:
    """Format a 95% confidence interval (e.g. [1.23, 4.56])."""
    return f"[{lower:.2f}, {upper:.2f}]"
