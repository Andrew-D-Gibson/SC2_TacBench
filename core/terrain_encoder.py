def terrain_encoder(
	terrain_height,
	pathing_grid,
	placement_grid=None,
	downsample_factor: int = 4,
	include_legend: bool = True,
	include_summary: bool = False,
	include_axes: bool = True,
) -> str:
	"""
	Convert SC2 terrain data into an LLM-friendly symbolic map.

	Args:
		terrain_height: 2D array-like (self.game_info.terrain_height.data)
		pathing_grid:   2D array-like (self.game_info.pathing_grid.data)
		placement_grid: Optional 2D array-like (self.game_info.placement_grid.data)
		downsample_factor: int (e.g., 2, 4, 8)
		include_legend: include symbol legend
		include_summary: include basic map summary

	Returns:
		str: formatted terrain representation
	"""

	# --- Convert to simple indexable structures ---
	height = terrain_height
	pathing = pathing_grid
	placement = placement_grid

	h = len(height)
	w = len(height[0])

	# --- Compute height normalization ---
	min_h = min(min(row) for row in height)
	max_h = max(max(row) for row in height)
	range_h = max(1, max_h - min_h)

	def height_to_symbol(h_val: int) -> str:
		"""Map height to symbolic tiers."""
		norm = (h_val - min_h) / range_h
		if norm < 0.33:
			return "."
		elif norm < 0.66:
			return "~"
		else:
			return "^"

	def cell_symbol(x: int, y: int) -> str:
		"""Combine terrain + pathing + placement into one symbol."""
		if not pathing[y][x]:
			return "#"

		if placement is not None and not placement[y][x]:
			return "x"  # walkable but not buildable

		return height_to_symbol(height[y][x])

	# --- Downsampling ---
	def downsample():
		ds_rows = []

		for y in range(0, h, downsample_factor):
			row_chars = []

			for x in range(0, w, downsample_factor):
				# Collect block
				block = []
				for dy in range(downsample_factor):
					for dx in range(downsample_factor):
						yy = y + dy
						xx = x + dx
						if yy < h and xx < w:
							block.append(cell_symbol(xx, yy))

				# Majority vote for symbol
				if not block:
					row_chars.append("?")
					continue

				# Priority: blocked > unbuildable > terrain
				if "#" in block:
					row_chars.append("#")
				elif "x" in block:
					row_chars.append("x")
				else:
					# Count terrain symbols
					counts = {}
					for b in block:
						counts[b] = counts.get(b, 0) + 1
					row_chars.append(max(counts, key=counts.get))

			ds_rows.append("".join(row_chars))

		return ds_rows

	ds_grid = downsample()

	# --- Axis labels ---
	def build_axes():
		num_rows = len(ds_grid)
		num_cols = len(ds_grid[0])

		max_y_coord = (num_rows - 1) * downsample_factor
		y_label_w = len(str(max_y_coord)) + 2  # digits + ": "

		labeled_rows = [
			f"{i * downsample_factor:>{y_label_w - 2}}: {row}"
			for i, row in enumerate(ds_grid)
		]

		max_x_coord = (num_cols - 1) * downsample_factor
		x_label_digits = len(str(max_x_coord))
		tick_interval = x_label_digits + 1

		x_row = ['.'] * (num_cols + x_label_digits)  # extra room so last label never truncates
		for col in range(0, num_cols, tick_interval):
			label = str(col * downsample_factor)
			for i, ch in enumerate(label):
				x_row[col + i] = ch

		x_axis = ' ' * y_label_w + ''.join(x_row).rstrip('.')

		return labeled_rows, x_axis

	# --- Summary stats ---
	def build_summary():
		total_cells = h * w
		blocked = sum(1 for y in range(h) for x in range(w) if not pathing[y][x])

		return (
			f"Map Size: {w}x{h}\n"
			f"Downsampled: {len(ds_grid[0])}x{len(ds_grid)} (factor={downsample_factor})\n"
		)

	# --- Legend ---
	def build_legend():
		legend = [
			"Legend:",
			". = low ground",
			"~ = medium ground",
			"^ = high ground",
			"# = unpathable",
		]
		if placement is not None:
			legend.append("x = walkable but not buildable")
		return "\n".join(legend)

	# --- Assemble output ---
	parts = []

	if include_summary:
		parts.append(build_summary())

	if include_legend:
		parts.append(build_legend())

	parts.append(f"Terrain (1 cell = {downsample_factor} game units):")

	if include_axes:
		labeled_rows, x_axis = build_axes()
		parts.append("\n".join(labeled_rows) + "\n" + x_axis)
	else:
		parts.append("\n".join(ds_grid))

	return "\n\n".join(parts)