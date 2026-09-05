"""
Module 5: Inference Engine
Direct pattern inference from input/output pairs.
Uses diagnostic classification to generate candidate programs efficiently.
"""
import sys
import os
import time
import math
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from substrate import *
from primitives import (
    PRIMITIVE_REGISTRY, ALL_PRIMITIVES, apply_primitive, apply_sequence,
    total_mdl, verify_program, solve_with_program,
    rotate, reflect_h, reflect_v, reflect_diag, transpose, translate_prim,
    recolor, swap_colors, palette_invert, color_to,    crop, pad, scale_uniform, scale_pattern, fill_holes, fill_interior,
    fill_by_component, draw_line, connect_adjacent,
    filter_color, select_largest, select_by_size, select_color_range,
    tile_grid, count_and_fill, sort_components,
    overlay, extend_to_symmetry, logical_or, logical_and, logical_xor,
    identity, pattern_tile, pattern_tile_custom, expand_shape, shrink_shape, recolor_component,
    _swap_colors_for_tile
)


def infer_color_mapping(inp, out):
    """Infer single global color mapping. Returns dict or None."""
    if grid_dims(inp) != grid_dims(out):
        return None
    ct = color_transition_matrix(inp, out)
    if ct is None:
        return None
    mapping = {}
    for src in range(10):
        for tgt in range(10):
            if ct[src][tgt] > 0:
                if src in mapping and mapping[src] != tgt:
                    return None
                mapping[src] = tgt
    return mapping if len(mapping) > 1 else None


def infer_color_mapping_for_pair(inp, out):
    """Infer color mapping for a single pair (may differ from global)."""
    if grid_dims(inp) != grid_dims(out):
        return None
    ct = color_transition_matrix(inp, out)
    if ct is None:
        return None
    mapping = {}
    for src in range(10):
        for tgt in range(10):
            if ct[src][tgt] > 0:
                if src in mapping and mapping[src] != tgt:
                    return None
                mapping[src] = tgt
    return mapping


def infer_single_color_change(inp, out):
    if grid_dims(inp) != grid_dims(out):
        return None
    ct = color_transition_matrix(inp, out)
    if ct is None:
        return None
    changes = [(s, t) for s in range(10) for t in range(10) if ct[s][t] > 0 and s != t]
    if len(changes) == 1:
        return changes[0]
    return None


def infer_translation(inp, out):
    if grid_dims(inp) != grid_dims(out):
        return None
    result = find_translation(inp, out)
    if result is None:
        return None
    dr, dc, score = result
    shifted = translate(inp, dr, dc, fill=background_color(inp))
    if grid_equals(shifted, out):
        return (dr, dc, score)
    return None


def try_scale_uniform(inp, out):
    """Try uniform scale. Returns (factor, scale_func) or None."""
    h_in, w_in = grid_dims(inp)
    h_out, w_out = grid_dims(out)
    if h_out > h_in and w_out > w_in and h_out % h_in == 0 and w_out % w_in == 0:
        fh, fw = h_out // h_in, w_out // w_in
        if fh == fw:
            result = scale_uniform(inp, factor=fh)
            if grid_equals(result, out):
                return (fh, "scale_uniform")
    return None


def try_tile(inp, out):
    """Try tiling. Returns (rows, cols) or None."""
    h_in, w_in = grid_dims(inp)
    h_out, w_out = grid_dims(out)
    if h_out > h_in and w_out > w_in and h_out % h_in == 0 and w_out % w_in == 0:
        fh, fw = h_out // h_in, w_out // w_in
        result = tile_grid(inp, rows=fh, cols=fw)
        if grid_equals(result, out):
            return (fh, fw)
    return None


def try_pattern_tile(inp, out):
    """Try tiling with various swap strategies."""
    h_in, w_in = grid_dims(inp)
    h_out, w_out = grid_dims(out)
    if h_in == 0 or w_in == 0 or h_out == 0 or w_out == 0:
        return None
    if h_out > h_in and w_out > w_in and h_out % h_in == 0 and w_out % w_in == 0:
        fh, fw = h_out // h_in, w_out // w_in
        swap_strategies = [
            ("color_swap", _swap_colors_for_tile(inp)),
            ("reflect_v", reflect_v(inp)),
            ("reflect_h", reflect_h(inp)),
            ("reflect_diag", reflect_diag(inp)),
            ("rotate_180", rotate(inp, 180)),
            ("transpose", transpose(inp)),
        ]
        # Only add rotations that preserve dimensions
        if grid_dims(rotate(inp, 90)) == (h_in, w_in):
            swap_strategies.append(("rotate_90", rotate(inp, 90)))
            swap_strategies.append(("rotate_270", rotate(inp, 270)))

        for swap_name, swapped_tile in swap_strategies:
            if grid_dims(swapped_tile) != (h_in, w_in):
                continue
            for swap_mode in ["row_alternate", "col_alternate", "checkerboard", "none"]:
                if swap_mode == "none" and swap_name != "color_swap":
                    continue
                result = [[0]*w_out for _ in range(h_out)]
                for tr in range(fh):
                    for tc in range(fw):
                        use_swap = (
                            swap_mode == "row_alternate" and tr % 2 == 1 or
                            swap_mode == "col_alternate" and tc % 2 == 1 or
                            swap_mode == "checkerboard" and (tr + tc) % 2 == 1
                        )
                        tile = swapped_tile if use_swap else inp
                        for r in range(h_in):
                            for c in range(w_in):
                                result[tr*h_in + r][tc*w_in + c] = tile[r][c]
                if grid_equals(result, out):
                    return (fh, fw, f"{swap_name}:{swap_mode}")
    return None


def try_recolor(inp, out):
    """Try recolor with global mapping."""
    mapping = infer_color_mapping(inp, out)
    if mapping:
        result = recolor(inp, mapping=mapping)
        if grid_equals(result, out):
            return mapping
    return None


def try_color_to(inp, out):
    """Try single color change."""
    change = infer_single_color_change(inp, out)
    if change:
        result = color_to(inp, src=change[0], tgt=change[1])
        if grid_equals(result, out):
            return change
    return None


def try_swap_colors(inp, out):
    """Try swapping two colors."""
    if grid_dims(inp) != grid_dims(out):
        return None
    ct = color_transition_matrix(inp, out)
    if ct is None:
        return None
    # Look for two-way swap: c1->c2 and c2->c1
    for c1 in range(10):
        for c2 in range(c1+1, 10):
            if ct[c1][c2] > 0 and ct[c2][c1] > 0:
                # Check if this is the ONLY transformation
                total_non_zero = sum(1 for s in range(10) for t in range(10) 
                                     if ct[s][t] > 0 and s != t)
                expected_changes = ct[c1][c2] + ct[c2][c1]
                if total_non_zero == expected_changes:
                    result = swap_colors(inp, c1=c1, c2=c2)
                    if grid_equals(result, out):
                        return (c1, c2)
    return None


def try_fill_holes(inp, out):
    """Try filling background holes with a new color."""
    if grid_dims(inp) != grid_dims(out):
        return None
    h, w = grid_dims(inp)
    bg = background_color(inp)
    new_cells = [(r, c) for r in range(h) for c in range(w)
                 if inp[r][c] == bg and out[r][c] != bg]
    if not new_cells:
        return None
    new_color = out[new_cells[0][0]][new_cells[0][1]]
    if all(out[r][c] == new_color for r, c in new_cells):
        result = fill_holes(inp, fill_color=new_color)
        if grid_equals(result, out):
            return new_color
    return None


def try_fill_interior(inp, out):
    """Try filling interior of shapes with border color."""
    if grid_dims(inp) != grid_dims(out):
        return None
    h, w = grid_dims(inp)
    bg = background_color(inp)
    new_cells = [(r, c) for r in range(h) for c in range(w)
                 if inp[r][c] == bg and out[r][c] != bg]
    if not new_cells:
        return None
    # Check if all new cells have the same color as some border color
    new_color = out[new_cells[0][0]][new_cells[0][1]]
    result = fill_interior(inp, fill_color=new_color, border_color=0)
    if grid_equals(result, out):
        return new_color
    # Try fill_by_component
    result = fill_by_component(inp)
    if grid_equals(result, out):
        return True
    return None


def try_crop(inp, out):
    """Try cropping. Returns bbox or None."""
    h_in, w_in = grid_dims(inp)
    h_out, w_out = grid_dims(out)
    if h_in >= h_out and w_in >= w_out and h_out > 0 and w_out > 0:
        for r1 in range(min(h_in - h_out + 1, 5)):
            for c1 in range(min(w_in - w_out + 1, 5)):
                r2, c2 = r1 + h_out - 1, c1 + w_out - 1
                if r2 < h_in and c2 < w_in:
                    result = crop(inp, bbox=[r1, c1, r2, c2])
                    if grid_equals(result, out):
                        return [r1, c1, r2, c2]
    return None


def try_pad(inp, out):
    """Try padding."""
    h_in, w_in = grid_dims(inp)
    h_out, w_out = grid_dims(out)
    if h_in <= h_out and w_in <= w_out:
        for fill in [0]:
            for pos in ["center", "topleft"]:
                result = pad(inp, target_h=h_out, target_w=w_out, fill=fill, position=pos)
                if grid_equals(result, out):
                    return {"fill": fill, "position": pos}
    return None


def try_extend_symmetry(inp, out):
    """Try extension to symmetry."""
    if grid_dims(inp) != grid_dims(out):
        return None
    for axis in ["horizontal", "vertical"]:
        result = extend_to_symmetry(inp, axis=axis)
        if grid_equals(result, out):
            return axis
    return None


def generate_candidates(task):
    """
    Main inference: for each primitive type, infer params from train[0],
    verify on ALL train pairs.
    Returns list of candidate programs.
    """
    train = task.get("train", [])
    if not train:
        return []
    
    inp0, out0 = train[0]["input"], train[0]["output"]
    h_in, w_in = grid_dims(inp0)
    h_out, w_out = grid_dims(out0)
    candidates = []
    
    # === Single primitive candidates ===
    
    # Identity
    if grid_equals(inp0, out0):
        candidates.append([("identity", {})])
    
    # Rotate
    for angle in [90, 180, 270]:
        if grid_equals(rotate(inp0, angle), out0):
            candidates.append([("rotate", {"angle": angle})])
    
    # Reflect
    if grid_equals(reflect_h(inp0), out0):
        candidates.append([("reflect_h", {})])
    if grid_equals(reflect_v(inp0), out0):
        candidates.append([("reflect_v", {})])
    if grid_equals(reflect_diag(inp0), out0):
        candidates.append([("reflect_diag", {})])
    if grid_equals(transpose(inp0), out0):
        candidates.append([("transpose", {})])
    
    # Translate
    trans = infer_translation(inp0, out0)
    if trans:
        candidates.append([("translate", {"dr": trans[0], "dc": trans[1]})])
    
    # Scale
    scale = try_scale_uniform(inp0, out0)
    if scale:
        candidates.append([("scale_uniform", {"factor": scale[0]})])
    
    # Tile
    tile = try_tile(inp0, out0)
    if tile:
        candidates.append([("tile", {"rows": tile[0], "cols": tile[1]})])
    
    # Pattern tile (checkerboard swap with various strategies)
    ptile = try_pattern_tile(inp0, out0)
    if ptile:
        swap_name, swap_mode = ptile[2].split(":")
        swap_fn_map = {
            "color_swap": _swap_colors_for_tile,
            "reflect_v": reflect_v,
            "reflect_h": reflect_h,
            "reflect_diag": reflect_diag,
            "rotate_180": lambda g: rotate(g, 180),
            "rotate_90": lambda g: rotate(g, 90),
            "rotate_270": lambda g: rotate(g, 270),
            "transpose": transpose,
        }
        swapped = swap_fn_map[swap_name](inp0)
        candidates.append([
            ("pattern_tile_custom", {"rows": ptile[0], "cols": ptile[1],
                                     "swap_mode": swap_mode, "swap_strategy": swap_name})
        ])
    
    # Pad
    pad_result = try_pad(inp0, out0)
    if pad_result:
        candidates.append([("pad", {"target_h": h_out, "target_w": w_out, "fill": pad_result["fill"], "position": pad_result["position"]})])
    
    # Crop
    crop_result = try_crop(inp0, out0)
    if crop_result:
        candidates.append([("crop", {"bbox": crop_result})])
    
    # Color operations
    mapping = try_recolor(inp0, out0)
    if mapping:
        candidates.append([("recolor", {"mapping": mapping})])
    
    swap = try_swap_colors(inp0, out0)
    if swap:
        candidates.append([("swap_colors", {"c1": swap[0], "c2": swap[1]})])
    
    single_change = try_color_to(inp0, out0)
    if single_change:
        candidates.append([("color_to", {"src": single_change[0], "tgt": single_change[1]})])
    
    # Fill holes
    fill_color = try_fill_holes(inp0, out0)
    if fill_color is not None:
        candidates.append([("fill_holes", {"fill_color": fill_color})])
    
    # Fill interior
    fill_inter = try_fill_interior(inp0, out0)
    if fill_inter is not None:
        if fill_inter is True:
            candidates.append([("fill_by_component", {})])
        else:
            candidates.append([("fill_interior", {"fill_color": fill_inter})])
    
    # Extend symmetry
    sym = try_extend_symmetry(inp0, out0)
    if sym:
        candidates.append([("extend_to_symmetry", {"axis": sym})])
    
    # Filter color
    if (h_in, w_in) == (h_out, w_out):
        bg = background_color(inp0)
        out_colors = set(all_colors(out0))
        if len(out_colors) == 1:
            keep = list(out_colors)[0]
            if keep != bg:
                candidates.append([("filter_color", {"keep": keep, "others": bg})])
    
    # Select largest
    if (h_in, w_in) == (h_out, w_out):
        if grid_equals(select_largest(inp0), out0):
            candidates.append([("select_largest", {})])
    
    # Palette invert
    if (h_in, w_in) == (h_out, w_out):
        if grid_equals(palette_invert(inp0), out0):
            candidates.append([("palette_invert", {"bg": 0})])
    
    # === 2-primitive compositions ===
    
    # Recolor + translate
    if (h_in, w_in) == (h_out, w_out):
        m = infer_color_mapping(inp0, out0)
        if m:
            recolored = recolor(inp0, mapping=m)
            t = infer_translation(recolored, out0)
            if t and (abs(t[0]) > 0 or abs(t[1]) > 0):
                candidates.append([
                    ("recolor", {"mapping": m}),
                    ("translate", {"dr": t[0], "dc": t[1]})
                ])
        
        # Translate + recolor
        trans = infer_translation(inp0, out0)
        if trans:
            shifted = translate(inp0, trans[0], trans[1], fill=background_color(inp0))
            m = infer_color_mapping(shifted, out0)
            if m:
                candidates.append([
                    ("translate", {"dr": trans[0], "dc": trans[1]}),
                    ("recolor", {"mapping": m})
                ])
    
    # Scale + recolor
    if h_out > h_in and w_out > w_in:
        scale = try_scale_uniform(inp0, out0)
        if scale:
            scaled = scale_uniform(inp0, factor=scale[0])
            m = infer_color_mapping(scaled, out0)
            if m:
                candidates.append([
                    ("scale_uniform", {"factor": scale[0]}),
                    ("recolor", {"mapping": m})
                ])
        # Tile + recolor
        tile = try_tile(inp0, out0)
        if tile:
            tiled = tile_grid(inp0, rows=tile[0], cols=tile[1])
            m = infer_color_mapping(tiled, out0)
            if m:
                candidates.append([
                    ("tile", {"rows": tile[0], "cols": tile[1]}),
                    ("recolor", {"mapping": m})
                ])
    
    # Fill holes + recolor
    if (h_in, w_in) == (h_out, w_out):
        for fill_c in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            filled = fill_holes(inp0, fill_color=fill_c)
            m = infer_color_mapping(filled, out0)
            if m and len(m) > 1:
                candidates.append([
                    ("fill_holes", {"fill_color": fill_c}),
                    ("recolor", {"mapping": m})
                ])
            elif grid_equals(filled, out0):
                candidates.append([("fill_holes", {"fill_color": fill_c})])
    
    # Rotate/Reflect + translate
    for reflect_fn, reflect_name in [(reflect_h, "reflect_h"), (reflect_v, "reflect_v")]:
        reflected = reflect_fn(inp0)
        if grid_dims(reflected) == (h_out, w_out):
            t = infer_translation(reflected, out0)
            if t:
                candidates.append([(reflect_name, {}), ("translate", {"dr": t[0], "dc": t[1]})])
    
    return candidates


class SearchResult:
    def __init__(self):
        self.program = None
        self.output = None
        self.confidence = 0.0
        self.search_time = 0.0
        self.engine = None
        self.n_evaluated = 0
        self.mdl_cost = 0

    def is_valid(self):
        return self.program is not None and self.output is not None


def verify_and_score(program, task):
    if verify_program(program, task):
        outputs = solve_with_program(task, program)
        if all(o is not None for o in outputs):
            return True, outputs, total_mdl(program)
    return False, None, math.inf


def solve_task(task, time_budget=20.0):
    """Main solver entry point."""
    start = time.time()
    n_eval = 0
    
    candidates = generate_candidates(task)
    best_result = SearchResult()
    best_mdl = math.inf
    
    for program in candidates:
        if time.time() - start > time_budget:
            break
        n_eval += 1
        is_valid, outputs, mdl = verify_and_score(program, task)
        if is_valid and mdl < best_mdl:
            best_mdl = mdl
            best_result.program = program
            best_result.output = outputs
            best_result.confidence = 1.0
            best_result.mdl_cost = mdl
            best_result.engine = "inference"
    
    if best_result.is_valid():
        best_result.search_time = time.time() - start
        best_result.n_evaluated = n_eval
        return best_result
    
    result = SearchResult()
    result.search_time = time.time() - start
    result.n_evaluated = n_eval
    result.engine = "failed"
    return result


if __name__ == "__main__":
    import json, glob
    
    task_files = sorted(glob.glob("/tmp/kilo/arc_data/training/*.json"))
    eval_files = sorted(glob.glob("/tmp/kilo/arc_data/evaluation/*.json"))
    all_files = task_files + eval_files
    
    start = time.time()
    solved = 0
    total = 0
    
    for tf in all_files[:30]:
        if time.time() - start > 300:
            print("Time limit")
            break
        with open(tf) as f:
            task = json.load(f)
        tid = os.path.basename(tf).replace(".json", "")
        
        result = solve_task(task, time_budget=8.0)
        total += 1
        
        if result.is_valid():
            test_outputs = result.output
            test_pairs = task.get("test", [])
            correct = 0
            for i, tp in enumerate(test_pairs):
                if "output" in tp and i < len(test_outputs) and test_outputs[i] is not None:
                    if grid_equals(test_outputs[i], tp["output"]):
                        correct += 1
            
            if correct == len(test_pairs):
                solved += 1
                status = "PASS"
            elif correct > 0:
                status = f"PARTIAL({correct}/{len(test_pairs)})"
            else:
                status = "WRONG"
        else:
            status = "FAIL"
        
        print(f"  {tid}: {status} | engine={result.engine} | evals={result.n_evaluated} | time={result.search_time:.2f}s")
    
    print(f"\nSolved: {solved}/{total} in {time.time()-start:.1f}s")
