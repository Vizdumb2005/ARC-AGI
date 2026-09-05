"""
Module 4: Search - Diagnostic-Guided Program Synthesis
Pure Python, deterministic. Uses diagnostic classification + object-centric
inference to generate candidate programs, then verifies against all train pairs.
"""
import sys
import os
import time
import math
from collections import deque, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from substrate import *
from primitives import (
    PRIMITIVE_REGISTRY, ALL_PRIMITIVES, apply_primitive, apply_sequence,
    total_mdl, verify_program, solve_with_program
)


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


class SearchContext:
    def __init__(self, task, time_budget=30.0):
        self.task = task
        self.train_pairs = task.get("train", [])
        self.test_inputs = [task["test"][i]["input"]
                           for i in range(len(task.get("test", [])))] if "test" in task else []
        self.time_budget = time_budget
        self.start_time = time.time()
        self.n_evaluated = 0
        self.timeout = False

    def elapsed(self):
        return time.time() - self.start_time
    def remaining(self):
        return max(0, self.time_budget - self.elapsed())
    def check_timeout(self):
        if self.elapsed() > self.time_budget:
            self.timeout = True
            return True
        return False


def generate_param_sets(prim_name, task, inp, out):
    """Generate focused parameter sets for a primitive on a specific pair."""
    h_in, w_in = grid_dims(inp)
    h_out, w_out = grid_dims(out)
    params = []

    if prim_name == "identity":
        params.append({})

    elif prim_name == "rotate":
        for angle in [0, 90, 180, 270]:
            params.append({"angle": angle})

    elif prim_name in ("reflect_h", "reflect_v", "reflect_diag", "transpose"):
        params.append({})

    elif prim_name == "translate":
        trans = find_translation(inp, out)
        if trans and trans[2] > 0:
            params.append({"dr": trans[0], "dc": trans[1]})
        for dr in [-3, -2, -1, 0, 1, 2, 3]:
            for dc in [-3, -1, 0, 1, 2, 3]:
                params.append({"dr": dr, "dc": dc})

    elif prim_name == "scale_uniform":
        if h_out > h_in and w_out > w_in:
            if h_out % h_in == 0 and w_out % w_in == 0:
                fh, fw = h_out // h_in, w_out // w_in
                if fh == fw:
                    params.append({"factor": fh})
                else:
                    # Try individual factors if close
                    if h_out % h_in == 0: params.append({"factor": fh})
                    if w_out % w_in == 0: params.append({"factor": fw})

    elif prim_name == "tile":
        if h_out >= h_in and w_out >= w_in:
            if h_out % h_in == 0 and w_out % w_in == 0:
                params.append({"rows": h_out // h_in, "cols": w_out // w_in})

    elif prim_name == "scale_pattern":
        if h_out > h_in and w_out > w_in:
            if h_out % h_in == 0 and w_out % w_in == 0:
                params.append({"rows": h_out // h_in, "cols": w_out // w_in})

    elif prim_name == "crop":
        params.append({})  # auto-crop
        if h_out <= h_in and w_out <= w_in:
            # Try all possible crop positions that match output dims
            for r1 in range(0, h_in - h_out + 1):
                for c1 in range(0, w_in - w_out + 1):
                    r2, c2 = r1 + h_out - 1, c1 + w_out - 1
                    if r2 < h_in and c2 < w_in:
                        params.append({"bbox": [r1, c1, r2, c2]})

    elif prim_name == "pad":
        if h_out >= h_in and w_out >= w_in:
            for pos in ["center", "topleft"]:
                params.append({"target_h": h_out, "target_w": w_out, "fill": 0, "position": pos})

    elif prim_name == "recolor":
        mapping = _infer_color_mapping(task.get("train", []))
        if mapping and len(mapping) > 1:
            params.append({"mapping": mapping})
            # Also try individual color changes
            for src in range(10):
                for tgt in range(10):
                    if src != tgt:
                        params.append({"mapping": {src: tgt}})

    elif prim_name == "swap_colors":
        if (h_in, w_in) == (h_out, w_out):
            ct = color_transition_matrix(inp, out)
            if ct:
                swaps = []
                for s1 in range(1, 10):
                    for s2 in range(s1+1, 10):
                        if ct[s1][s2] > 0 and ct[s2][s1] > 0:
                            params.append({"c1": s1, "c2": s2})
                # Also try common swaps
                for c1 in range(1, 10):
                    for c2 in range(1, 10):
                        if c1 != c2:
                            params.append({"c1": c1, "c2": c2})

    elif prim_name == "color_to":
        if (h_in, w_in) == (h_out, w_out):
            ct = color_transition_matrix(inp, out)
            if ct:
                for src in range(10):
                    for tgt in range(10):
                        if ct[src][tgt] > 0:
                            params.append({"src": src, "tgt": tgt})

    elif prim_name == "palette_invert":
        params.append({"bg": 0})

    elif prim_name == "fill_holes":
        for fc in range(1, 10):
            params.append({"fill_color": fc})

    elif prim_name == "draw_line":
        bg = background_color(inp)
        fg = [(r, c) for r in range(h_in) for c in range(w_in) if inp[r][c] != bg]
        if len(fg) >= 2:
            for i in range(min(3, len(fg))):
                for j in range(i+1, min(5, len(fg))):
                    for color in range(1, 10):
                        params.append({"p1": [fg[i][0], fg[i][1]],
                                      "p2": [fg[j][0], fg[j][1]], "color": color})
        for r in range(h_in):
            for c in range(w_in):
                if inp[r][c] == bg:
                    for color in range(1, 10):
                        params.append({"p1": [0, 0], "p2": [r, c], "color": color})
                    break
            if params:
                break

    elif prim_name == "connect_adjacent":
        for color in range(1, 10):
            params.append({"color": color})

    elif prim_name == "filter_color":
        bg = background_color(inp)
        for c in range(0, 10):
            params.append({"keep": c, "others": bg})

    elif prim_name == "select_largest":
        params.append({})

    elif prim_name == "select_by_size":
        params.append({"min_area": 1, "max_area": 999})
        params.append({"min_area": 0, "max_area": 5})

    elif prim_name == "select_color_range":
        for lo in [1, 2]:
            for hi in [5, 9]:
                params.append({"lo": lo, "hi": hi})

    elif prim_name == "tile":
        for rows in [2, 3, 4]:
            for cols in [2, 3, 4]:
                params.append({"rows": rows, "cols": cols})

    elif prim_name == "count_and_fill":
        for cc in range(1, 10):
            for fc in range(1, 10):
                params.append({"count_color": cc, "fill_color": fc})

    elif prim_name == "sort_components":
        for direction in ["vertical", "horizontal"]:
            for by in ["area", "color"]:
                params.append({"direction": direction, "by": by})

    elif prim_name == "overlay":
        for mc in range(1, 10):
            for tc in range(0, 10):
                params.append({"mask_color": mc, "target_color": tc})

    elif prim_name == "extend_to_symmetry":
        params.append({"axis": "horizontal"})
        params.append({"axis": "vertical"})

    elif prim_name == "logical_or":
        for c1 in range(1, 10):
            for c2 in range(c1+1, 10):
                params.append({"colors": [c1, c2]})

    elif prim_name == "logical_xor":
        for c in range(1, 10):
            params.append({"color": c})

    else:
        params.append({})

    # Deduplicate
    seen = set()
    unique = []
    for p in params:
        key = str(sorted(p.items())) if isinstance(p, dict) else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique if unique else None


def _infer_color_mapping(train_pairs):
    mapping = {}
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        if grid_dims(inp) != grid_dims(out):
            return None
        ct = color_transition_matrix(inp, out)
        if ct is None:
            return None
        pair_map = {}
        for src in range(10):
            for tgt in range(10):
                if ct[src][tgt] > 0:
                    if src in pair_map and pair_map[src] != tgt:
                        return None
                    pair_map[src] = tgt
        if not mapping:
            mapping = pair_map
        else:
            for k, v in pair_map.items():
                if k in mapping and mapping[k] != v:
                    return None
    return mapping if mapping else None


def try_single_primitives(task, ctx):
    """Phase 1: Try all single-primitive candidates."""
    train = task["train"]
    if not train:
        return SearchResult()
    
    best_result = SearchResult()
    best_mdl = math.inf
    inp0, out0 = train[0]["input"], train[0]["output"]
    
    for prim_name in ALL_PRIMITIVES:
        if ctx.check_timeout():
            break
        prim = PRIMITIVE_REGISTRY[prim_name]
        param_sets = generate_param_sets(prim_name, task, inp0, out0)
        if param_sets is None:
            continue
        for params in param_sets[:15]:
            ctx.n_evaluated += 1
            program = [(prim_name, params)]
            if verify_program(program, task):
                outputs = solve_with_program(task, program)
                if all(o is not None for o in outputs):
                    mdl = total_mdl(program)
                    if mdl < best_mdl:
                        best_mdl = mdl
                        best_result.program = program
                        best_result.output = outputs
                        best_result.confidence = 1.0
                        best_result.mdl_cost = mdl
                        best_result.engine = "single_primitive"
    
    return best_result


def try_two_primitive_compositions(task, ctx):
    """Phase 2: Try targeted 2-primitive compositions."""
    train = task["train"]
    if not train:
        return SearchResult()
    
    best_result = SearchResult()
    best_mdl = math.inf
    inp0, out0 = train[0]["input"], train[0]["output"]
    h_in, w_in = grid_dims(inp0)
    h_out, w_out = grid_dims(out0)
    
    # Generate 1-primitive candidates for first position
    first_candidates = []
    for prim_name in ALL_PRIMITIVES:
        param_sets = generate_param_sets(prim_name, task, inp0, out0)
        if param_sets is None:
            continue
        for params in param_sets[:3]:
            first_candidates.append((prim_name, params))
    
    for prim1_name, params1 in first_candidates:
        if ctx.check_timeout():
            break
        
        # Apply first primitive to create intermediate
        try:
            inter = apply_primitive(inp0, prim1_name, params1)
        except:
            continue
        if inter is None:
            continue
        
        # Generate second candidates based on intermediate -> output
        for prim2_name in ALL_PRIMITIVES:
            if ctx.check_timeout():
                break
            param_sets2 = generate_param_sets(prim2_name, task, inter, out0)
            if param_sets2 is None:
                continue
            for params2 in param_sets2[:3]:
                ctx.n_evaluated += 1
                program = [(prim1_name, params1), (prim2_name, params2)]
                if verify_program(program, task):
                    outputs = solve_with_program(task, program)
                    if all(o is not None for o in outputs):
                        mdl = total_mdl(program)
                        if mdl < best_mdl:
                            best_mdl = mdl
                            best_result.program = program
                            best_result.output = outputs
                            best_result.confidence = 1.0
                            best_result.mdl_cost = mdl
                            best_result.engine = "two_composition"
    
    return best_result


def try_object_centric(task, ctx):
    """Phase 3: Object-centric inference.
    Extract objects, match input->output, infer transformations."""
    train = task["train"]
    if not train:
        return SearchResult()
    
    best_result = SearchResult()
    best_mdl = math.inf
    
    # For each pair, try to find a consistent object transformation
    for pair in train:
        if ctx.check_timeout():
            break
        inp, out = pair["input"], pair["output"]
        h_in, w_in = grid_dims(inp)
        h_out, w_out = grid_dims(out)
        
        comps_in = connected_components(inp, connectivity=4)
        comps_out = connected_components(out, connectivity=4)
        
        if not comps_in or not comps_out:
            continue
        
        # Try: each input component maps to an output component via (dr, dc) translation
        # Find the most common translation
        translations = Counter()
        for ci in comps_in:
            for co in comps_out:
                if ci["color"] != co["color"]:
                    continue
                # Find best translation
                best_trans = None
                best_overlap = 0
                out_cells = set(co["cells"])
                for dr in range(-h_in, h_in):
                    if ctx.check_timeout():
                        break
                    for dc in range(-w_in, w_in):
                        overlap = sum(1 for (r,c) in ci["cells"]
                                     if (r+dr, c+dc) in out_cells)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_trans = (dr, dc)
                if best_trans and best_overlap > 0:
                    translations[best_trans] += 1
        
        if translations:
            # Most common translation
            best_trans, count = translations.most_common(1)[0]
            if count >= len(comps_in):
                ctx.n_evaluated += 1
                program = [("translate", {"dr": best_trans[0], "dc": best_trans[1]})]
                if verify_program(program, task):
                    outputs = solve_with_program(task, program)
                    if all(o is not None for o in outputs):
                        mdl = total_mdl(program)
                        if mdl < best_mdl:
                            best_mdl = mdl
                            best_result.program = program
                            best_result.output = outputs
                            best_result.confidence = 1.0
                            best_result.mdl_cost = mdl
                            best_result.engine = "object_centric"
        
        # If translation didn't work, try matching with color changes
        if ctx.remaining() > 5 and not best_result.is_valid():
            # Try: translate + recolor, or recolor + translate
            if translations:
                best_trans, _ = translations.most_common(1)[0]
                for src_c in range(1, 10):
                    for tgt_c in range(0, 10):
                        if src_c == tgt_c:
                            continue
                        ctx.n_evaluated += 1
                        program = [
                            ("translate", {"dr": best_trans[0], "dc": best_trans[1]}),
                            ("color_to", {"src": src_c, "tgt": tgt_c})
                        ]
                        if verify_program(program, task):
                            outputs = solve_with_program(task, program)
                            if all(o is not None for o in outputs):
                                mdl = total_mdl(program)
                                if mdl < best_mdl:
                                    best_mdl = mdl
                                    best_result.program = program
                                    best_result.output = outputs
                                    best_result.confidence = 1.0
                                    best_result.mdl_cost = mdl
                                    best_result.engine = "object_centric"
    
    return best_result


def try_pattern_matching(task, ctx):
    """Phase 4: Pattern-based inference from common ARC patterns."""
    train = task["train"]
    if not train:
        return SearchResult()
    
    best_result = SearchResult()
    best_mdl = math.inf
    inp0, out0 = train[0]["input"], train[0]["output"]
    h_in, w_in = grid_dims(inp0)
    h_out, w_out = grid_dims(out0)
    
    # Pattern 1: Tile with checkerboard color swap
    # e.g., input 2x2 -> output 6x6 where alternating tiles have swapped colors
    if h_out > h_in and w_out > w_in:
        if h_out % h_in == 0 and w_out % w_in == 0:
            fh, fw = h_out // h_in, w_out // w_in
            # Check if the first tile matches the input
            sub = extract_subgrid(out0, 0, 0, h_in-1, w_in-1)
            if grid_equals(sub, inp0):
                # Check if the diagonal tile is color-swapped
                sub2 = extract_subgrid(out0, h_in, w_in, 2*h_in-1, 2*w_in-1)
                if sub2 != [[]]:
                    # Check color swap
                    swapped = _swap_all_colors(inp0)
                    if grid_equals(sub2, swapped):
                        # Try composing: tile + swap colors in alternating tiles
                        # Actually, let's try: tile, then recolor based on position
                        ctx.n_evaluated += 1
                        # For this specific pattern, we need custom logic
                        result = _tile_with_swap(inp0, fh, fw)
                        if result and grid_equals(result, out0):
                            # Verify on all pairs
                            all_match = True
                            for pair in train:
                                r = _tile_with_swap(pair["input"], fh, fw)
                                if r is None or not grid_equals(r, pair["output"]):
                                    all_match = False
                                    break
                            if all_match:
                                outputs = [_tile_with_swap(t["input"], fh, fw)
                                          for t in task.get("test", [])]
                                if all(o is not None for o in outputs):
                                    # Represent as tile + conditional recolor
                                    program = [("scale_pattern", {"rows": fh, "cols": fw})]
                                    best_mdl = total_mdl(program)
                                    best_result.program = program
                                    best_result.output = outputs
                                    best_result.confidence = 1.0
                                    best_result.engine = "pattern_match"
    
    # Pattern 2: Fill holes
    if (h_in, w_in) == (h_out, w_out):
        # Check if output is input with holes filled
        bg = background_color(inp0)
        # Find 0s in input that are non-0 in output (same position)
        new_cells = []
        for r in range(h_in):
            for c in range(w_in):
                if inp0[r][c] == bg and out0[r][c] != bg and out0[r][c] != bg:
                    new_cells.append((r, c, out0[r][c]))
        
        if len(new_cells) > 0:
            new_color = new_cells[0][2]
            same_color = all(nc[2] == new_color for nc in new_cells)
            if same_color:
                ctx.n_evaluated += 1
                program = [("fill_holes", {"fill_color": new_color})]
                if verify_program(program, task):
                    outputs = solve_with_program(task, program)
                    if all(o is not None for o in outputs):
                        mdl = total_mdl(program)
                        if mdl < best_mdl:
                            best_mdl = mdl
                            best_result.program = program
                            best_result.output = outputs
                            best_result.confidence = 1.0
                            best_result.engine = "pattern_match"
    
    return best_result


def _swap_all_colors(g):
    """Swap all non-zero colors (simple swap: max+cmin-c)."""
    colors = sorted(set(c for r in range(len(g)) for c in range(len(g[0])) if g[r][c] != 0))
    if len(colors) < 2:
        return copy_grid(g)
    cmin, cmax = colors[0], colors[-1]
    result = copy_grid(g)
    for r in range(len(result)):
        for c in range(len(result[0])):
            v = result[r][c]
            if v != 0 and cmin <= v <= cmax:
                result[r][c] = cmin + cmax - v
    return result


def _tile_with_swap(g, fh, fw):
    """Tile grid fh x fw with alternating color swap per tile."""
    h, w = grid_dims(g)
    rh, rw = h * fh, w * fw
    result = [[0] * rw for _ in range(rh)]
    swapped = _swap_all_colors(g)
    
    for tr in range(fh):
        for tc in range(fw):
            if (tr + tc) % 2 == 0:
                tile = g
            else:
                tile = swapped
            for r in range(h):
                for c in range(w):
                    result[tr * h + r][tc * w + c] = tile[r][c]
    return result


def solve_task(task, time_budget=20.0):
    """
    Multi-phase solver: single primitives -> compositions -> object-centric -> pattern matching.
    Returns SearchResult with best verified program.
    """
    ctx = SearchContext(task, time_budget=time_budget)
    
    # Phase 1: Single primitives
    result = try_single_primitives(task, ctx)
    if result.is_valid():
        result.search_time = ctx.elapsed()
        result.n_evaluated = ctx.n_evaluated
        return result
    
    # Phase 2: Two-primitive compositions
    if ctx.remaining() > 5:
        result2 = try_two_primitive_compositions(task, ctx)
        if result2.is_valid():
            result2.search_time = ctx.elapsed()
            result2.n_evaluated = ctx.n_evaluated
            return result2
    
    # Phase 3: Object-centric
    if ctx.remaining() > 5:
        result3 = try_object_centric(task, ctx)
        if result3.is_valid():
            result3.search_time = ctx.elapsed()
            result3.n_evaluated = ctx.n_evaluated
            return result3
    
    # Phase 4: Pattern matching
    if ctx.remaining() > 5:
        result4 = try_pattern_matching(task, ctx)
        if result4.is_valid():
            result4.search_time = ctx.elapsed()
            result4.n_evaluated = ctx.n_evaluated
            return result4
    
    # Failed
    result = SearchResult()
    result.search_time = ctx.elapsed()
    result.n_evaluated = ctx.n_evaluated
    result.engine = "failed"
    return result


if __name__ == "__main__":
    import json, glob
    
    task_files = sorted(glob.glob("/tmp/kilo/arc_data/training/*.json"))
    
    start = time.time()
    solved = 0
    total = 0
    
    for tf in task_files[:30]:
        if time.time() - start > 300:
            print("Time limit reached")
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
