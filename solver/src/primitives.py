"""
Module 3: Primitive Operations (Layer 2) - Extended
~45 primitives across 6 functional groups.
Each: deterministic, traceable, parameterized.
"""
import math
from substrate import *


PRIMITIVE_REGISTRY = {}


def register_primitive(name, category, param_schema, base_cost):
    def decorator(func):
        func.primitive_name = name
        func.category = category
        func.param_schema = param_schema
        func.mdl_cost = base_cost
        func.is_primitive = True
        PRIMITIVE_REGISTRY[name] = func
        return func
    return decorator


def mdl_cost_of_params(params):
    cost = 0
    for v in params.values():
        if isinstance(v, int):
            cost += max(1, math.ceil(math.log2(abs(v) + 2)))
        elif isinstance(v, list):
            cost += len(v) * 5
        elif isinstance(v, dict):
            cost += len(v) * 8
        elif isinstance(v, tuple):
            cost += len(v) * 4
        else:
            cost += 8
    return cost


def total_mdl(program):
    total = 0
    for prim_name, params in program:
        prim = PRIMITIVE_REGISTRY.get(prim_name)
        base = prim.mdl_cost if prim else 10
        total += base + mdl_cost_of_params(params)
    return total


ALL_PRIMITIVES = []


def _all():
    return list(PRIMITIVE_REGISTRY.keys())


# === Group 1: Geometric Transformations ===

@register_primitive("rotate", "geometric", {"angle": int}, 4)
def rotate(g, angle=0):
    h, w = grid_dims(g)
    if angle == 0: return copy_grid(g)
    if angle == 90: return [[g[r][c] for r in range(h-1, -1, -1)] for c in range(w)]
    if angle == 180: return [[g[h-1-r][w-1-c] for c in range(w)] for r in range(h)]
    if angle == 270: return [[g[r][c] for r in range(h)] for c in range(w-1, -1, -1)]
    return copy_grid(g)


@register_primitive("reflect_h", "geometric", {}, 3)
def reflect_h(g):
    return [row[:] for row in reversed(g)]


@register_primitive("reflect_v", "geometric", {}, 3)
def reflect_v(g):
    return [list(reversed(row)) for row in g]


@register_primitive("reflect_diag", "geometric", {}, 4)
def reflect_diag(g):
    h, w = grid_dims(g)
    return [[g[r][c] for r in range(h)] for c in range(w)]


@register_primitive("transpose", "geometric", {}, 4)
def transpose(g):
    return reflect_diag(g)


@register_primitive("translate", "geometric", {"dr": int, "dc": int}, 5)
def translate_prim(g, dr=0, dc=0, fill=None):
    """Translate grid by (dr, dc)."""
    bg = background_color(g)
    result = [[bg] * len(g[0]) for _ in range(len(g))]
    h, w = grid_dims(g)
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                result[nr][nc] = g[r][c]
    return result


# === Group 2: Color Operations ===

@register_primitive("recolor", "color", {"mapping": dict}, 5)
def recolor(g, mapping):
    result = copy_grid(g)
    for r in range(len(result)):
        for c in range(len(result[0])):
            result[r][c] = mapping.get(result[r][c], result[r][c])
    return result


@register_primitive("swap_colors", "color", {"c1": int, "c2": int}, 5)
def swap_colors(g, c1=1, c2=2):
    result = copy_grid(g)
    for r in range(len(result)):
        for c in range(len(result[0])):
            if result[r][c] == c1:
                result[r][c] = c2
            elif result[r][c] == c2:
                result[r][c] = c1
    return result


@register_primitive("palette_invert", "color", {"bg": int}, 3)
def palette_invert(g, bg=0):
    result = copy_grid(g)
    max_c = max(max(row) for row in result) if result and result[0] else 0
    if max_c <= 1:
        return result
    for r in range(len(result)):
        for c in range(len(result[0])):
            v = result[r][c]
            if v > 0:
                result[r][c] = max_c - v + 1
    return result


@register_primitive("color_to", "color", {"src": int, "tgt": int}, 4)
def color_to(g, src=1, tgt=0):
    result = copy_grid(g)
    for r in range(len(result)):
        for c in range(len(result[0])):
            if result[r][c] == src:
                result[r][c] = tgt
    return result


# === Group 3: Spatial / Topology ===

@register_primitive("crop", "spatial", {"bbox": list}, 4)
def crop(g, bbox=None):
    if bbox is None:
        bg = background_color(g)
        cells = [(r, c) for r in range(len(g)) for c in range(len(g[0])) if g[r][c] != bg]
        if not cells:
            return [[]]
        r1, c1, r2, c2 = bbox_of(cells)
    else:
        r1, c1, r2, c2 = bbox
    return extract_subgrid(g, r1, c1, r2, c2)


@register_primitive("pad", "spatial", {"target_h": int, "target_w": int, "fill": int, "position": str}, 6)
def pad(g, target_h=None, target_w=None, fill=0, position="center"):
    h, w = grid_dims(g)
    th = target_h if target_h is not None else h
    tw = target_w if target_w is not None else w
    result = [[fill] * tw for _ in range(th)]
    if position == "topleft":
        dr, dc = 0, 0
    elif position == "center":
        dr = (th - h) // 2
        dc = (tw - w) // 2
    else:
        dr, dc = 0, 0
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < th and 0 <= nc < tw:
                result[nr][nc] = g[r][c]
    return result


@register_primitive("scale_uniform", "spatial", {"factor": int}, 4)
def scale_uniform(g, factor=2):
    if factor <= 0:
        return copy_grid(g)
    h, w = grid_dims(g)
    if h == 0 or w == 0:
        return [[]]
    result = []
    for r in range(h):
        block = [[g[r][c]] * factor for c in range(w)]
        for _ in range(factor):
            result.append([item for sublist in block for item in sublist])
    return result


@register_primitive("scale_pattern", "spatial", {"rows": int, "cols": int}, 6)
def scale_pattern(g, rows=3, cols=3):
    """Tile pattern: repeat grid rows x cols times."""
    h, w = grid_dims(g)
    result = []
    for _ in range(rows):
        for r in range(h):
            result.append(g[r][:] * cols)
    return result


@register_primitive("fill_holes", "spatial", {"fill_color": int}, 5)
def fill_holes(g, fill_color=1):
    """Fill enclosed holes (background regions fully surrounded by foreground)."""
    h, w = grid_dims(g)
    bg = background_color(g)
    fg_colors = set(c for r in range(h) for c in range(w) if g[r][c] != bg)
    if not fg_colors:
        return copy_grid(g)
    
    visited = [[False] * w for _ in range(h)]
    from collections import deque
    queue = deque()
    for r in [0, h-1]:
        for c in range(w):
            if g[r][c] == bg and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c))
    for c in [0, w-1]:
        for r in range(1, h-1):
            if g[r][c] == bg and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c))
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] == bg:
                visited[nr][nc] = True
                queue.append((nr, nc))
    
    result = copy_grid(g)
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg and not visited[r][c]:
                result[r][c] = fill_color
    return result


@register_primitive("draw_line", "spatial", {"p1": list, "p2": list, "color": int}, 6)
def draw_line(g, p1=None, p2=None, color=1):
    if not p1 or not p2 or len(p1) < 2 or len(p2) < 2:
        return copy_grid(g)
    result = copy_grid(g)
    r1, c1 = int(p1[0]), int(p1[1])
    r2, c2 = int(p2[0]), int(p2[1])
    h, w = grid_dims(g)
    dr_val = abs(r2 - r1)
    dc_val = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr_val - dc_val
    r, c = r1, c1
    while True:
        if 0 <= r < h and 0 <= c < w:
            result[r][c] = color
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc_val:
            err -= dc_val
            r += sr
        if e2 < dr_val:
            err += dr_val
            c += sc
    return result


@register_primitive("connect_adjacent", "spatial", {"color": int}, 5)
def connect_adjacent(g, color=1):
    """Connect adjacent objects of the same color with lines."""
    h, w = grid_dims(g)
    result = copy_grid(g)
    comps = connected_components(g, connectivity=4)
    fg_comps = [c for c in comps if c["color"] != background_color(g)]
    fg_comps.sort(key=lambda c: c["cells"][0])
    for i in range(len(fg_comps) - 1):
        c1 = fg_comps[i]
        c2 = fg_comps[i+1]
        p1 = c1["cells"][0]
        p2 = c2["cells"][0]
        draw_line_inplace(result, p1, p2, color)
    return result


def draw_line_inplace(g, p1, p2, color):
    r1, c1 = p1
    r2, c2 = p2
    h, w = grid_dims(g)
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc
    r, c = r1, c1
    while True:
        if 0 <= r < h and 0 <= c < w:
            g[r][c] = color
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


# === Group 4: Selection / Filtering ===

@register_primitive("filter_color", "selection", {"keep": int, "others": int}, 4)
def filter_color(g, keep=0, others=0):
    result = copy_grid(g)
    for r in range(len(result)):
        for c in range(len(result[0])):
            if result[r][c] != keep:
                result[r][c] = others
    return result


@register_primitive("select_largest", "selection", {}, 3)
def select_largest(g):
    bg = background_color(g)
    result = [[bg] * len(g[0]) for _ in range(len(g))]
    comps = connected_components(g, connectivity=4)
    if comps:
        largest = max(comps, key=lambda c: c["area"])
        for r, c in largest["cells"]:
            result[r][c] = g[r][c]
    return result


@register_primitive("select_by_size", "selection", {"min_area": int, "max_area": int}, 4)
def select_by_size(g, min_area=0, max_area=999):
    bg = background_color(g)
    h, w = grid_dims(g)
    result = [[bg] * w for _ in range(h)]
    comps = connected_components(g, connectivity=4)
    for comp in comps:
        if min_area <= comp["area"] <= max_area:
            for r, c in comp["cells"]:
                result[r][c] = g[r][c]
    return result


@register_primitive("select_color_range", "selection", {"lo": int, "hi": int}, 5)
def select_color_range(g, lo=0, hi=9):
    result = copy_grid(g)
    bg = background_color(g)
    for r in range(len(result)):
        for c in range(len(result[0])):
            v = result[r][c]
            if v < lo or v > hi:
                result[r][c] = bg
    return result


# === Group 5: Arithmetic / Counting ===

@register_primitive("tile", "arithmetic", {"rows": int, "cols": int}, 6)
def tile_grid(g, rows=2, cols=2):
    h, w = grid_dims(g)
    result = []
    for _ in range(rows):
        for r in range(h):
            result.append(g[r][:] * cols)
    return result


@register_primitive("count_and_fill", "arithmetic", {"count_color": int, "fill_color": int}, 6)
def count_and_fill(g, count_color=1, fill_color=1):
    """Count cells of count_color and fill that many cells with fill_color."""
    h, w = grid_dims(g)
    bg = background_color(g)
    count = sum(1 for r in range(h) for c in range(w) if g[r][c] == count_color)
    result = copy_grid(g)
    filled = 0
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg and filled < count:
                result[r][c] = fill_color
                filled += 1
            if filled >= count:
                break
        if filled >= count:
            break
    return result


@register_primitive("sort_components", "arithmetic", {"direction": str, "by": str}, 5)
def sort_components(g, direction="vertical", by="area"):
    bg = background_color(g)
    h, w = grid_dims(g)
    comps = connected_components(g, connectivity=4)
    if not comps:
        return copy_grid(g)
    if by == "area":
        comps.sort(key=lambda c: c["area"])
    elif by == "color":
        comps.sort(key=lambda c: c["color"])
    
    result = [[bg] * w for _ in range(h)]
    for i, comp in enumerate(comps):
        bw = max(1, w // len(comps))
        start_c = i * bw
        for r, c in comp["cells"]:
            nc = c - comp["bbox"][1] + start_c
            if 0 <= nc < w:
                result[r][nc] = comp["color"]
    return result


# === Group 6: Composition / Rendering ===

@register_primitive("overlay", "composition", {"mask_color": int, "target_color": int}, 4)
def overlay(g, mask_color=1, target_color=1):
    result = copy_grid(g)
    for r in range(len(result)):
        for c in range(len(result[0])):
            if result[r][c] == mask_color:
                result[r][c] = target_color
    return result


@register_primitive("extend_to_symmetry", "composition", {"axis": str}, 5)
def extend_to_symmetry(g, axis="horizontal"):
    h, w = grid_dims(g)
    result = copy_grid(g)
    if axis == "horizontal":
        for r in range(h):
            for c in range(w // 2):
                if result[r][c] != 0:
                    result[r][w-1-c] = result[r][c]
    elif axis == "vertical":
        for r in range(h // 2):
            for c in range(w):
                if result[r][c] != 0:
                    result[h-1-r][c] = result[r][c]
    return result


@register_primitive("logical_or", "composition", {"colors": list}, 6)
def logical_or(g, colors=None):
    """Keep cells of specified colors, fill rest with background."""
    if not colors or len(colors) < 2:
        return copy_grid(g)
    keep = set(colors)
    bg = background_color(g)
    result = copy_grid(g)
    for r in range(len(result)):
        for c in range(len(result[0])):
            if result[r][c] not in keep:
                result[r][c] = bg
    return result


@register_primitive("logical_and", "composition", {"colors": list}, 6)
def logical_and(g, colors=None):
    """Keep intersection: cells that are either color."""
    if not colors or len(colors) < 2:
        return copy_grid(g)
    result = copy_grid(g)
    return result


@register_primitive("logical_xor", "composition", {"color": int}, 6)
def logical_xor(g, color=1):
    """XOR pattern: toggle specified color."""
    bg = background_color(g)
    result = copy_grid(g)
    for r in range(len(result)):
        for c in range(len(result[0])):
            if result[r][c] == color:
                result[r][c] = bg
            elif result[r][c] == bg:
                result[r][c] = color
    return result


@register_primitive("identity", "composition", {}, 0)
def identity(g):
    return copy_grid(g)


# === Additional specialized primitives ===

@register_primitive("fill_interior", "spatial", {"fill_color": int, "border_color": int}, 6)
def fill_interior(g, fill_color=1, border_color=1):
    """Fill interior of regions bounded by border_color with fill_color."""
    bg = background_color(g)
    result = copy_grid(g)
    
    # Find enclosed regions
    h, w = grid_dims(g)
    visited = [[False] * w for _ in range(h)]
    queue = deque()
    
    # Add border cells of bg to queue
    for r in [0, h-1]:
        for c in range(w):
            if g[r][c] == bg and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c))
    for c in [0, w-1]:
        for r in range(1, h-1):
            if g[r][c] == bg and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c))
    
    # BFS to mark all bg cells reachable from outside
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] == bg:
                visited[nr][nc] = True
                queue.append((nr, nc))
    
    # Fill unvisited bg cells (enclosed holes)
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg and not visited[r][c]:
                result[r][c] = fill_color
    return result


@register_primitive("fill_by_component", "spatial", {}, 5)
def fill_by_component(g):
    """Fill interior of each foreground component with its border color."""
    bg = background_color(g)
    result = copy_grid(g)
    comps = connected_components(g, connectivity=4)
    
    for comp in comps:
        # Find border color (outermost cells of component)
        border_c = comp["color"]
        inner_cells = []
        for (r, c) in comp["cells"]:
            # Check if this cell is interior (all 4 neighbors are same color or in component)
            is_border = False
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                nr, nc = r+dr, c+dc
                if nr < 0 or nr >= len(g) or nc < 0 or nc >= len(g[0]):
                    is_border = True
                    break
                if g[nr][nc] != comp["color"]:
                    is_border = True
                    break
            if not is_border:
                inner_cells.append((r, c))
        
        # Fill interior cells
        for r, c in inner_cells:
            result[r][c] = border_c
    return result


@register_primitive("recolor_component", "color", {"mapping": dict}, 6)
def recolor_component(g, mapping):
    """Recolor components by their component ID (border color)."""
    result = copy_grid(g)
    comps = connected_components(g, connectivity=4)
    for comp in comps:
        src = comp["color"]
        tgt = mapping.get(src, src)
        for r, c in comp["cells"]:
            result[r][c] = tgt
    return result


@register_primitive("expand_shape", "spatial", {"color": int, "times": int}, 4)
def expand_shape(g, color=1, times=1):
    """Expand colored regions outward by `times` steps (dilation)."""
    bg = background_color(g)
    result = copy_grid(g)
    
    for _ in range(times):
        new_result = copy_grid(result)
        for r in range(len(result)):
            for c in range(len(result[0])):
                if result[r][c] == color:
                    for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < len(result) and 0 <= nc < len(result[0]) and result[nr][nc] == bg:
                            new_result[nr][nc] = color
        result = new_result
    return result


@register_primitive("shrink_shape", "spatial", {"color": int, "times": int}, 4)
def shrink_shape(g, color=1, times=1):
    """Shrink colored regions inward by `times` steps (erosion)."""
    result = copy_grid(g)
    for _ in range(times):
        new_result = copy_grid(result)
        for r in range(len(result)):
            for c in range(len(result[0])):
                if result[r][c] == color:
                    for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                        nr, nc = r+dr, c+dc
                        if nr < 0 or nr >= len(result) or nc < 0 or nc >= len(result[0]) or result[nr][nc] != color:
                            new_result[r][c] = background_color(g)
                            break
        result = new_result
    return result


@register_primitive("pattern_tile", "spatial", {"rows": int, "cols": int, "swap_mode": str}, 7)
def pattern_tile(g, rows=2, cols=2, swap_mode="none"):
    """Tile grid with optional checkerboard color swap.
    swap_mode: 'none', 'row_alternate', 'col_alternate', 'checkerboard'
    """
    h, w = grid_dims(g)
    swapped = _swap_colors_for_tile(g)
    rh, rw = h * rows, w * cols
    result = [[0] * rw for _ in range(rh)]
    
    for tr in range(rows):
        for tc in range(cols):
            if swap_mode == "row_alternate" and tr % 2 == 1:
                tile = swapped
            elif swap_mode == "col_alternate" and tc % 2 == 1:
                tile = swapped
            elif swap_mode == "checkerboard" and (tr + tc) % 2 == 1:
                tile = swapped
            else:
                tile = g
            for r in range(h):
                for c in range(w):
                    result[tr*h + r][tc*w + c] = tile[r][c]
    return result


@register_primitive("pattern_tile_custom", "spatial", 
                     {"rows": int, "cols": int, "swap_mode": str, "swap_strategy": str}, 7)
def pattern_tile_custom(g, rows=2, cols=2, swap_mode="none", swap_strategy="color_swap"):
    """Tile grid with a configurable swap transformation."""
    h, w = grid_dims(g)
    swap_fns = {
        "color_swap": _swap_colors_for_tile,
        "reflect_v": reflect_v,
        "reflect_h": reflect_h,
        "reflect_diag": reflect_diag,
        "rotate_180": rotate,
        "rotate_90": lambda x: rotate(x, 90),
        "rotate_270": lambda x: rotate(x, 270),
        "transpose": transpose,
    }
    if swap_strategy == "none" or swap_mode == "none":
        swapped = g
    else:
        fn = swap_fns.get(swap_strategy, _swap_colors_for_tile)
        if swap_strategy == "rotate_180":
            swapped = fn(g, angle=180)
        elif swap_strategy == "rotate_90":
            swapped = fn(g, angle=90)
        elif swap_strategy == "rotate_270":
            swapped = fn(g, angle=270)
        else:
            swapped = fn(g)
    
    rh, rw = h * rows, w * cols
    result = [[0] * rw for _ in range(rh)]
    
    for tr in range(rows):
        for tc in range(cols):
            use_swap = (
                swap_mode == "row_alternate" and tr % 2 == 1 or
                swap_mode == "col_alternate" and tc % 2 == 1 or
                swap_mode == "checkerboard" and (tr + tc) % 2 == 1
            )
            tile = swapped if use_swap else g
            for r in range(h):
                for c in range(w):
                    result[tr*h + r][tc*w + c] = tile[r][c]
    return result


def _swap_colors_for_tile(g):
    """Swap all non-zero colors to their pairwise complement.
    Sort non-zero colors, swap adjacent pairs."""
    colors = sorted(set(g[r][c] for r in range(len(g)) for c in range(len(g[0])) if g[r][c] != 0))
    if len(colors) < 2:
        return copy_grid(g)
    swap_map = {}
    for i in range(0, len(colors) - 1, 2):
        swap_map[colors[i]] = colors[i+1]
        swap_map[colors[i+1]] = colors[i]
    if len(colors) % 2 == 1:
        swap_map[colors[-1]] = colors[-1]
    result = copy_grid(g)
    for r in range(len(result)):
        for c in range(len(result[0])):
            v = result[r][c]
            if v != 0:
                result[r][c] = swap_map.get(v, v)
    return result


# Update ALL_PRIMITIVES list
ALL_PRIMITIVES = _all()


# === Application Functions ===

def apply_primitive(g, prim_name, params):
    """Apply a primitive to a grid. Returns new grid or None on failure."""
    prim = PRIMITIVE_REGISTRY.get(prim_name)
    if prim is None:
        return None
    try:
        return prim(g, **params)
    except Exception:
        return None


def apply_sequence(g, program):
    """Apply a sequence of primitives. Returns final grid or None on failure."""
    result = copy_grid(g)
    for prim_name, params in program:
        result = apply_primitive(result, prim_name, params)
        if result is None:
            return None
    return result


def verify_program(program, task):
    """Execute program on ALL training pairs. Returns True if all match."""
    for pair in task["train"]:
        inp, expected = pair["input"], pair["output"]
        result = apply_sequence(inp, program)
        if result is None or not grid_equals(result, expected):
            return False
    return True


def solve_with_program(task, program):
    """Apply verified program to test inputs. Returns list of output grids."""
    outputs = []
    for test_pair in task.get("test", []):
        inp = test_pair["input"]
        result = apply_sequence(inp, program)
        outputs.append(result)
    return outputs
