"""House colour palette, for consistent use across the paper figures.

The base colours are saturated by design, which is often too intense for a
dense visualisation with several colours side by side. Use `tint()` to soften
one toward white, or `shade()` to darken it, before dropping it into a plot;
how much depends on the figure, so there is no single right amount.

Usage:
    from src.eval.colors import BLUE, YELLOW, tint
    majority_color = tint(BLUE, 0.5)
    tail_color = tint(YELLOW, 0.15)
"""

# Neutral
ANTHRACITE = "#555555"

# Warm tones
YELLOW = "#ffc900"
RED = "#ea3b07"

# Cool tones
TEAL = "#00a0aa"
BLUE = "#004282"

PALETTE = {
    "anthracite": ANTHRACITE,
    "yellow": YELLOW,
    "red": RED,
    "teal": TEAL,
    "blue": BLUE,
}


def tint(hex_color: str, amount: float = 0.4) -> str:
    """Lighten `hex_color` toward white by `amount` (0 = unchanged, 1 = white).
    Reach for this instead of picking an arbitrary new hex code when a colour
    needs to be softer."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = round(r + (255 - r) * amount)
    g = round(g + (255 - g) * amount)
    b = round(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def shade(hex_color: str, amount: float = 0.4) -> str:
    """Darken `hex_color` toward black by `amount` (0 = unchanged, 1 = black),
    the opposite of tint, for when a colour reads too light rather than too
    strong (e.g. YELLOW on a white background)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = round(r * (1 - amount))
    g = round(g * (1 - amount))
    b = round(b * (1 - amount))
    return f"#{r:02x}{g:02x}{b:02x}"
