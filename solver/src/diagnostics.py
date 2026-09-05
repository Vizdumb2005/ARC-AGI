"""
Module 1: Diagnostics Pipeline
Computes a 7-dimensional diagnostic vector for each input/output pair.
All deterministic and inspectable.
"""
from collections import Counter
from substrate import *


def diagnose_pair(g_in, g_out):
    """
    Produce a diagnostic signature dict for one train/test pair.
    Answers: what kind of transformation occurred?
    """
    h_in, w_in = grid_dims(g_in)
    h_out, w_out = grid_dims(g_out)
    diag = {}

    # 1. Dimensional transformation vector
    dh, dw = h_out - h_in, w_out - w_in
    diag["dim"] = {
        "h_in": h_in, "w_in": w_in,
        "h_out": h_out, "w_out": w_out,
        "dh": dh, "dw": dw,
        "scale_ratio": (h_out / h_in if h_in > 0 else 0, w_out / w_in if w_in > 0 else 0),
        "size_change": "identity" if dh == 0 and dw == 0 else
                       "downscale" if h_out <= h_in and w_out <= w_in and (dh != 0 or dw != 0) else
                       "upscale" if h_out >= h_in and w_out >= w_in and (dh != 0 or dw != 0) else
                       "variable"
    }

    # 2. Color frequency shift matrix (if same size)
    if (h_in, w_in) == (h_out, w_out):
        diag["color_transitions"] = color_transition_matrix(g_in, g_out)
    else:
        diag["color_transitions"] = None

    # 3. Pixel difference mask (if same size)
    diff = diff_mask(g_in, g_out)
    if diff:
        changes = sum(sum(row) for row in diff)
        total = h_in * w_in
        diag["pixel_diff"] = {
            "changed": changes,
            "total": total,
            "frac_changed": changes / total if total > 0 else 0
        }
    else:
        diag["pixel_diff"] = None

    # 3b. Changed cells details (if same size)
    if (h_in, w_in) == (h_out, w_out):
        diag["changed_cells"] = changed_cells(g_in, g_out) or []
    else:
        diag["changed_cells"] = []

    # 4. Neighborhood dependency analysis (if same size)
    if (h_in, w_in) == (h_out, w_out):
        local_deps = 0
        local_total = 0
        for r in range(h_in):
            for c in range(w_in):
                if g_in[r][c] != g_out[r][c]:
                    hood_in = neighborhood(g_in, r, c, radius=1)
                    hood_out = neighborhood(g_out, r, c, radius=1)
                    local_total += 1
                    match = True
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < h_in and 0 <= nc < w_in:
                                if g_in[nr][nc] != g_out[nr][nc]:
                                    match = False
                    if match:
                        local_deps += 1
        diag["neighborhood"] = {
            "local_deps": local_deps,
            "local_total": local_total,
            "is_cellular_automaton": local_deps > local_total * 0.5 if local_total > 0 else False
        }
    else:
        diag["neighbourhood"] = {"is_ca": False}

    # 5. Symmetry scores
    diag["symmetry_in"] = symmetry_score_continuous(g_in)
    diag["symmetry_out"] = symmetry_score_continuous(g_out)

    # 6. Translation / offset mapping
    if (h_in, w_in) == (h_out, w_out):
        trans = find_translation(g_in, g_out)
        diag["translation"] = {"dr": trans[0], "dc": trans[1], "score": trans[2]} if trans else None
    else:
        diag["translation"] = None

    # 7. Topological metrics
    comps_in = connected_components(g_in, connectivity=4)
    comps_out = connected_components(g_out, connectivity=4)

    def topo(comps, h, w):
        return {
            "n_components": len(comps),
            "total_area": sum(c["area"] for c in comps),
            "colors": sorted(set(c["color"] for c in comps)),
            "avg_area": sum(c["area"] for c in comps) / len(comps) if comps else 0,
            "grid_fill": sum(c["area"] for c in comps) / (h * w) if h * w > 0 else 0
        }

    diag["topology_in"] = topo(comps_in, h_in, w_in)
    diag["topology_out"] = topo(comps_out, h_out, w_out)

    # Component count changes
    diag["comp_delta"] = {
        "in_count": len(comps_in),
        "out_count": len(comps_out),
        "delta": len(comps_out) - len(comps_in)
    }

    return diag


def diagnose_task(task):
    """Run diagnostics on all train pairs, return list of signature dicts."""
    results = []
    for i, pair in enumerate(task["train"]):
        d = diagnose_pair(pair["input"], pair["output"])
        d["index"] = i
        results.append(d)
    return results


def summarize_task(task):
    """High-level summary of task transformation patterns."""
    diags = diagnose_task(task)
    summary = {
        "n_train": len(diags),
        "n_test": len(task.get("test", [])),
        "size_changes": [],
        "color_preserving": [],
        "component_deltas": [],
        "symmetry_gains": [],
    }
    for d in diags:
        summary["size_changes"].append(d["dim"]["size_change"])
        if d["pixel_diff"]:
            summary["color_preserving"].append(d["pixel_diff"]["frac_changed"] < 0.5)
        summary["component_deltas"].append(d["comp_delta"]["delta"])

    # Overall pattern
    from collections import Counter as Cntr
    sc = Cntr(summary["size_changes"])
    summary["dominant_size_change"] = sc.most_common(1)[0][0] if sc else "unknown"

    return summary


if __name__ == "__main__":
    # Test on downloaded data
    import json, os, glob, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    task_files = sorted(glob.glob("/tmp/kilo/arc_data/training/*.json"))[:10]
    for tf in task_files:
        with open(tf) as f:
            task = json.load(f)
        tid = os.path.basename(tf).replace(".json", "")
        summary = summarize_task(task)
        d0 = diagnose_task(task)[0]
        pc = d0['pixel_diff']
        pc_str = f"frac_changed={pc['frac_changed']:.2f}" if pc else "diff_size"
        print(f"  {tid}: {summary['n_train']} pairs, size_change={summary['dominant_size_change']}, "
              f"{pc_str}, "
              f"comps: {d0['comp_delta']['in_count']}->{d0['comp_delta']['out_count']}")
    
    print("\nDiagnostics module working!")
