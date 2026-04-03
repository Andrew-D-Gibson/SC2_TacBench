"""
terrain_encoder.py — SC2 terrain → LLM-friendly ASCII map.

Public API
----------
build_terrain_grid(terrain_height, pathing_grid, placement_grid, downsample_factor)
    → list[str]   (raw downsampled rows; row 0 = game y=0, i.e. south edge)

format_terrain_grid(ds_grid, downsample_factor, *, flip_y, overlays, ...)
    → str         (formatted string ready for the LLM prompt)

terrain_encoder(...)
    → str         (convenience: build + format in one call; original public API)
"""


# ── Symbol helpers ─────────────────────────────────────────────────────────────

def _height_to_symbol(h_val: int, min_h: int, range_h: int) -> str:
    norm = (h_val - min_h) / range_h
    if norm < 0.33:
        return "."
    if norm < 0.66:
        return "~"
    return "^"


def _cell_symbol(x: int, y: int, height, pathing, placement, min_h: int, range_h: int) -> str:
    if not pathing[y][x]:
        return "#"
    if placement is not None and not placement[y][x]:
        return "x"
    return _height_to_symbol(height[y][x], min_h, range_h)


# ── Build step (expensive — cache the result) ──────────────────────────────────

def build_terrain_grid(
    terrain_height,
    pathing_grid,
    placement_grid=None,
    downsample_factor: int = 4,
) -> list[str]:
    """
    Downsample SC2 terrain data into a list of symbol rows.

    Row 0 corresponds to game y=0 (the south / bottom edge of the SC2 map).
    Call format_terrain_grid() to flip, overlay labels, and produce a string.
    """
    height    = terrain_height
    pathing   = pathing_grid
    placement = placement_grid

    h = len(height)
    w = len(height[0])

    min_h   = min(min(row) for row in height)
    max_h   = max(max(row) for row in height)
    range_h = max(1, max_h - min_h)

    df = downsample_factor
    ds_rows: list[str] = []

    for y in range(0, h, df):
        row_chars: list[str] = []
        for x in range(0, w, df):
            block: list[str] = []
            for dy in range(df):
                for dx in range(df):
                    yy, xx = y + dy, x + dx
                    if yy < h and xx < w:
                        block.append(_cell_symbol(xx, yy, height, pathing, placement, min_h, range_h))

            if not block:
                row_chars.append("?")
                continue

            if "#" in block:
                row_chars.append("#")
            elif "x" in block:
                row_chars.append("x")
            else:
                counts: dict[str, int] = {}
                for b in block:
                    counts[b] = counts.get(b, 0) + 1
                row_chars.append(max(counts, key=counts.get))

        ds_rows.append("".join(row_chars))

    return ds_rows


# ── Format step (cheap — runs every frame when overlays change) ────────────────

def format_terrain_grid(
    ds_grid: list[str],
    downsample_factor: int = 4,
    *,
    flip_y: bool = True,
    overlays: list[tuple[str, float, float]] | None = None,
    include_legend: bool = True,
    include_summary: bool = False,
    include_axes: bool = True,
    orig_h: int = 0,
    orig_w: int = 0,
) -> str:
    """
    Format a pre-built downsampled grid into an LLM-ready string.

    Parameters
    ----------
    ds_grid          : output of build_terrain_grid()
    downsample_factor: the same factor used when building
    flip_y           : if True (default), reverse rows so north is at the top,
                       matching the SC2 in-game perspective
    overlays         : list of (label_char, game_x, game_y) tuples to stamp on
                       the grid before formatting; useful for unit cluster markers
    include_legend   : print symbol key
    include_summary  : print map dimensions
    include_axes     : print y-labels on the left and x-ruler at the bottom
    orig_h / orig_w  : original map dimensions for the summary line
    """
    df        = downsample_factor
    num_rows  = len(ds_grid)
    num_cols  = len(ds_grid[0]) if ds_grid else 0

    # Work on a mutable character grid so overlays can be stamped.
    grid: list[list[str]] = [list(row) for row in ds_grid]

    # Apply overlays (in game-coordinate space, before the flip).
    # Friendly clusters are written first; enemy clusters written second so they
    # win if they share a cell (enemy = higher tactical urgency to see).
    if overlays:
        for label, gx, gy in overlays:
            col = max(0, min(num_cols - 1, round(gx / df)))
            row = max(0, min(num_rows - 1, round(gy / df)))
            grid[row][col] = label[0]

    # Convert back to strings.
    rows: list[str] = ["".join(r) for r in grid]

    # Flip: row 0 of ds_grid is game y=0 (south). After reversal, the last row
    # (highest game y, north) appears at the top of the printed output —
    # matching the SC2 screen orientation.
    if flip_y:
        rows = list(reversed(rows))

    # ── Axis labels ────────────────────────────────────────────────────────────

    def build_axes() -> tuple[list[str], str]:
        max_y_coord = (num_rows - 1) * df
        y_label_w   = len(str(max_y_coord)) + 2  # digits + ": "

        labeled: list[str] = []
        for i, row in enumerate(rows):
            # After flip, row i corresponds to game y = (num_rows-1-i)*df
            game_y = ((num_rows - 1 - i) * df) if flip_y else (i * df)
            labeled.append(f"{game_y:>{y_label_w - 2}}: {row}")

        max_x_coord    = (num_cols - 1) * df
        x_label_digits = len(str(max_x_coord))
        tick_interval  = x_label_digits + 1

        x_row = ["."] * (num_cols + x_label_digits)
        for col in range(0, num_cols, tick_interval):
            label = str(col * df)
            for k, ch in enumerate(label):
                x_row[col + k] = ch

        x_axis = " " * y_label_w + "".join(x_row).rstrip(".")
        return labeled, x_axis

    # ── Legend ─────────────────────────────────────────────────────────────────

    def build_legend() -> str:
        lines = [
            "\nLegend:",
            ". = low ground",
            "~ = medium ground",
            "^ = high ground",
            "# = unpathable",
        ]
        if overlays:
            lines.append("A-Z = your unit groups")
            lines.append("1-9 = enemy clusters")
        return "\n".join(lines)

    # ── Summary ────────────────────────────────────────────────────────────────

    def build_summary() -> str:
        return (
            f"Map Size: {orig_w}x{orig_h}\n"
            f"Downsampled: {num_cols}x{num_rows} (factor={df})\n"
        )

    # ── Assemble ───────────────────────────────────────────────────────────────

    parts: list[str] = []

    if include_summary:
        parts.append(build_summary())
    if include_legend:
        parts.append(build_legend())

    parts.append(f"Terrain (1 cell = {df} game units):")

    if include_axes:
        labeled_rows, x_axis = build_axes()
        parts.append("\n".join(labeled_rows) + "\n" + x_axis)
    else:
        parts.append("\n".join(rows))

    return "\n\n".join(parts)


# ── Convenience wrapper (original public API) ──────────────────────────────────

def terrain_encoder(
    terrain_height,
    pathing_grid,
    placement_grid=None,
    downsample_factor: int = 4,
    include_legend: bool = True,
    include_summary: bool = False,
    include_axes: bool = True,
    flip_y: bool = True,
    overlays: list[tuple[str, float, float]] | None = None,
) -> str:
    """Build and format in one call. Suitable when caching is not needed."""
    h      = len(terrain_height)
    w      = len(terrain_height[0])
    ds     = build_terrain_grid(terrain_height, pathing_grid, placement_grid, downsample_factor)
    return format_terrain_grid(
        ds,
        downsample_factor,
        flip_y=flip_y,
        overlays=overlays,
        include_legend=include_legend,
        include_summary=include_summary,
        include_axes=include_axes,
        orig_h=h,
        orig_w=w,
    )
