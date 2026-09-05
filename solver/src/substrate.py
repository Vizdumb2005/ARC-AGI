"""
Module 0: Minimal Grid Substrate for ARC-AGI-2
Pure Python, no external dependencies.
Core abstraction: a grid is a finite rectangular array of integers 0-9.
"""
import json
from collections import deque, Counter


COLOR_CHARS = {
    0: ".", 1: "B", 2: "R", 3: "G", 4: "Y",
    5: "X", 6: "P", 7: "O", 8: "C", 9: "V"
}


# === Grid Core Operations ===

def load_grid(data):
    return [row[:] for row in data]


def grid_dims(g):
    if not g:
        return (0, 0)
    return (len(g), len(g[0]))


def grid_equals(g1, g2):
    if grid_dims(g1) != grid_dims(g2):
        return False
    return all(row1 == row2 for row1, row2 in zip(g1, g2))


def make_grid(h, w, fill=0):
    return [[fill] * w for _ in range(h)]


def copy_grid(g):
    return [row[:] for row in g]


def get_cell(g, r, c):
    if 0 <= r < len(g) and 0 <= c < len(g[0]):
        return g[r][c]
    return None


def set_cell(g, r, c, val):
    g[r][c] = val


# === Color Analysis ===

def color_freq(g):
    counts = Counter()
    for row in g:
        counts.update(row)
    return dict(counts)


def all_colors(g):
    return sorted(color_freq(g).keys())


def is_background(color):
    return color == 0


def dominant_color(g):
    counts = color_freq(g)
    return max(counts, key=counts.get) if counts else 0


def background_color(g):
    counts = color_freq(g)
    return max(counts, key=counts.get) if counts else 0


# === Difference Analysis ===

def diff_mask(g1, g2):
    if grid_dims(g1) != grid_dims(g2):
        return None
    h, w = grid_dims(g1)
    return [[g1[r][c] != g2[r][c] for c in range(w)] for r in range(h)]


def size_delta(g_in, g_out):
    h_in, w_in = grid_dims(g_in)
    h_out, w_out = grid_dims(g_out)
    return (h_out - h_in, w_out - w_in)


def color_transition_matrix(g_in, g_out):
    h1, w1 = grid_dims(g_in)
    h2, w2 = grid_dims(g_out)
    if (h1, w1) != (h2, w2):
        return None
    mat = [[0] * 10 for _ in range(10)]
    for r in range(h1):
        for c in range(w1):
            mat[g_in[r][c]][g_out[r][c]] += 1
    return mat


def changed_cells(g1, g2):
    if grid_dims(g1) != grid_dims(g2):
        return None
    changes = []
    for r in range(len(g1)):
        for c in range(len(g1[0])):
            if g1[r][c] != g2[r][c]:
                changes.append((r, c, g1[r][c], g2[r][c]))
    return changes


# === Neighborhood analysis ===

def neighborhood(g, r, c, radius=1):
    h, w = grid_dims(g)
    result = []
    for dr in range(-radius, radius + 1):
        row = []
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                row.append(g[nr][nc])
            else:
                row.append(-1)
        result.append(row)
    return result


# === Symmetry Checks ===

def h_symmetric(g):
    h, w = grid_dims(g)
    for r in range(h):
        for c in range(w // 2):
            if g[r][c] != g[r][w - 1 - c]:
                return False
    return True


def v_symmetric(g):
    h, w = grid_dims(g)
    for r in range(h // 2):
        for c in range(w):
            if g[r][c] != g[h - 1 - r][c]:
                return False
    return True


def rot180_symmetric(g):
    h, w = grid_dims(g)
    for r in range(h):
        for c in range(w):
            if g[r][c] != g[h - 1 - r][w - 1 - c]:
                return False
    return True


def symmetry_scores(g):
    return {
        "horizontal": h_symmetric(g),
        "vertical": v_symmetric(g),
        "rot180": rot180_symmetric(g),
    }


def symmetry_score_continuous(g):
    h, w = grid_dims(g)
    scores = {}

    total = h * (w // 2)
    matches = sum(1 for r in range(h) for c in range(w // 2) if g[r][c] == g[r][w - 1 - c])
    scores["horizontal"] = matches / total if total > 0 else 1.0

    total = (h // 2) * w
    matches = sum(1 for r in range(h // 2) for c in range(w) if g[r][c] == g[h - 1 - r][c])
    scores["vertical"] = matches / total if total > 0 else 1.0

    total = h * w
    matches = sum(1 for r in range(h) for c in range(w) if g[r][c] == g[h - 1 - r][w - 1 - c])
    scores["rot180"] = matches / total if total > 0 else 1.0

    return scores


# === Translation / Offset ===

def translate(g, dr, dc, fill=0):
    h, w = grid_dims(g)
    result = [[fill] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                result[nr][nc] = g[r][c]
    return result


def find_translation(g1, g2):
    """Find (dr, dc) best mapping g1->g2. Returns (dr, dc, score) or None."""
    h, w = grid_dims(g1)
    if grid_dims(g1) != grid_dims(g2):
        return None
    best_offset = (0, 0)
    best_score = -1
    for dr in range(-h + 1, h):
        for dc in range(-w + 1, w):
            score = 0
            total = 0
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        total += 1
                        if g1[nr][nc] == g2[r][c]:
                            score += 1
            if total > 0 and score > best_score:
                best_score = score
                best_offset = (dr, dc)
    return (best_offset[0], best_offset[1], best_score)


# === Connected Components ===

def connected_components(g, connectivity=4):
    h, w = grid_dims(g)
    visited = [[False] * w for _ in range(h)]
    components = []

    if connectivity == 4:
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    else:
        neighbors = [(dr, dc) for dr in range(-1, 2) for dc in range(-1, 2) if (dr, dc) != (0, 0)]

    for r0 in range(h):
        for c0 in range(w):
            if visited[r0][c0]:
                continue
            color = g[r0][c0]
            if color == 0:
                visited[r0][c0] = True
                continue
            cells = []
            queue = deque([(r0, c0)])
            visited[r0][c0] = True
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] == color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
            if cells:
                r1 = min(c[0] for c in cells)
                c1 = min(c[1] for c in cells)
                r2 = max(c[0] for c in cells)
                c2 = max(c[1] for c in cells)
                components.append({
                    "color": color,
                    "cells": cells,
                    "bbox": (r1, c1, r2, c2),
                    "area": len(cells)
                })
    return components


def color_regions(g):
    h, w = grid_dims(g)
    regions = {}
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            if v not in regions:
                regions[v] = []
            regions[v].append((r, c))
    return regions


# === Shape / Object Analysis ===

def bbox_of(cells):
    r1 = min(r for r, c in cells)
    c1 = min(c for r, c in cells)
    r2 = max(r for r, c in cells)
    c2 = max(c for r, c in cells)
    return (r1, c1, r2, c2)


def extract_subgrid(g, r1, c1, r2, c2):
    return [row[c1:c2 + 1] for row in g[r1:r2 + 1]]


def render_ascii(g, symbols=None):
    if symbols is None:
        symbols = COLOR_CHARS
    lines = []
    for row in g:
        line = " ".join(symbols.get(v, str(v)) for v in row)
        lines.append(line)
    return "\n".join(lines)


def render_pair(inp, out, label=""):
    parts = []
    if label:
        parts.append(f"=== {label} ===")
    parts.append("INPUT:")
    parts.append(render_ascii(inp))
    parts.append("OUTPUT:")
    parts.append(render_ascii(out))
    return "\n".join(parts)


# === Trace Format ===

def make_trace_entry(entry_type, data, confidence=None):
    return {"type": entry_type, "data": data, "confidence": confidence}


def trace_str(trace):
    lines = []
    for entry in trace:
        if isinstance(entry, dict):
            t = entry.get("type", "?")
            d = entry.get("data", {})
            c = entry.get("confidence", None)
            lines.append(f"[{t}] conf={c} {d}")
        else:
            lines.append(str(entry))
    return "\n".join(lines)


if __name__ == "__main__":
    test_grid = [[1, 2, 0], [0, 1, 2], [2, 0, 1]]
    assert grid_dims(test_grid) == (3, 3)
    assert color_freq(test_grid) == {1: 3, 2: 3, 0: 3}
    assert not h_symmetric(test_grid)
    cc = connected_components(test_grid, connectivity=4)
    assert len(cc) >= 1
    shifted = translate(test_grid, 1, 0)
    assert grid_dims(shifted) == (3, 3)
    print("Substrate self-tests passed!")
    print(render_ascii(test_grid))
