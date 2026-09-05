"""
Module 2: Multi-View Representations
Maintains 6 competing representations of each grid in parallel.
No single representation is assumed correct; MDL scores rank them.
"""
from substrate import *
import copy


class GridView:
    """Base class: every view can render back to a grid (invertibility)."""
    
    @classmethod
    def from_grid(cls, g):
        raise NotImplementedError
    
    def to_grid(self):
        raise NotImplementedError
    
    def mdl_cost(self):
        """Description length in bits."""
        raise NotImplementedError
    
    def info(self):
        """Summary for debugging."""
        return {}


class RawPixelView(GridView):
    """Layer 1: Raw pixel matrix. Zero assumptions."""
    
    @classmethod
    def from_grid(cls, g):
        obj = cls()
        obj.grid = copy_grid(g)
        return obj
    
    def to_grid(self):
        return copy_grid(self.grid)
    
    def mdl_cost(self):
        h, w = grid_dims(self.grid)
        return h * w * 4  # ~4 bits per cell (0-9 needs 4 bits)
    
    def info(self):
        return {"type": "raw", "dims": grid_dims(self.grid)}


class ConnectedComponentView(GridView):
    """Layer 2: Connected components as regions."""
    
    @classmethod
    def from_grid(cls, g):
        obj = cls()
        obj.grid = copy_grid(g)
        obj.components_4 = connected_components(g, connectivity=4)
        obj.components_8 = connected_components(g, connectivity=8)
        obj.regions = color_regions(g)
        return obj
    
    def to_grid(self):
        return copy_grid(self.grid)
    
    def mdl_cost(self):
        # Cost = count + per-component bbox + color + area
        c = len(self.components_4) * 10  # ~10 bits per component
        for comp in self.components_4:
            c += 20  # bbox (4 coords * 5 bits) + color (4 bits) + area (8 bits)
        return c
    
    def info(self):
        return {
            "type": "cc",
            "n_components_4": len(self.components_4),
            "n_components_8": len(self.components_8),
            "colors": sorted(self.regions.keys()),
        }


class ColorMaskView(GridView):
    """Layer 3: Per-color bitmask regions."""
    
    @classmethod
    def from_grid(cls, g):
        obj = cls()
        obj.grid = copy_grid(g)
        obj.masks = color_regions(g)
        return obj
    
    def to_grid(self):
        return copy_grid(self.grid)
    
    def mdl_cost(self):
        h, w = grid_dims(self.grid)
        c = len(self.masks) * 4  # color per mask
        for color, cells in self.masks.items():
            c += len(cells) * (self._bits_needed(h) + self._bits_needed(w))
        return c
    
    @staticmethod
    def _bits_needed(n):
        import math
        return max(1, math.ceil(math.log2(n + 1)))
    
    def info(self):
        return {
            "type": "colormask",
            "colors": {c: len(cells) for c, cells in self.masks.items()}
        }


class BoundingBoxView(GridView):
    """Layer 4: List of object tuples (bbox, color, area)."""
    
    @classmethod
    def from_grid(cls, g):
        obj = cls()
        obj.grid = copy_grid(g)
        comps = connected_components(g, connectivity=4)
        obj.objects = []
        for comp in comps:
            r1, c1, r2, c2 = comp["bbox"]
            obj.objects.append({
                "color": comp["color"],
                "bbox": (r1, c1, r2, c2),
                "area": comp["area"],
                "cells": comp["cells"]
            })
        return obj
    
    def to_grid(self):
        return copy_grid(self.grid)
    
    def mdl_cost(self):
        c = len(self.objects) * 32  # header
        for obj in self.objects:
            c += 4 * 5 + 4 + 8  # 4 coords(5b) + color(4b) + area(8b)
        return c
    
    def info(self):
        return {
            "type": "bbox",
            "n_objects": len(self.objects),
            "objects": [{"color": o["color"], "bbox": o["bbox"], "area": o["area"]} for o in self.objects]
        }


class RelationalGraphView(GridView):
    """Layer 5: Objects as nodes, spatial relations as edges."""
    
    @classmethod
    def from_grid(cls, g):
        obj = cls()
        obj.grid = copy_grid(g)
        comps = connected_components(g, connectivity=4)
        obj.nodes = []
        obj.edges = []
        
        # Create nodes (skip background)
        for i, comp in enumerate(comps):
            obj.nodes.append({
                "id": i,
                "color": comp["color"],
                "bbox": comp["bbox"],
                "area": comp["area"],
                "center": (sum(r for r,c in comp["cells"])/len(comp["cells"]),
                          sum(c for r,c in comp["cells"])/len(comp["cells"]))
            })
        
        # Create edges (spatial relations)
        for i in range(len(obj.nodes)):
            for j in range(i+1, len(obj.nodes)):
                n1, n2 = obj.nodes[i], obj.nodes[j]
                relations = compute_relations(n1, n2)
                if relations:
                    obj.edges.append({"src": i, "tgt": j, "relations": relations})
        return obj
    
    def to_grid(self):
        return copy_grid(self.grid)
    
    def mdl_cost(self):
        c = len(self.nodes) * 40
        c += len(self.edges) * 20
        for e in self.edges:
            c += len(e["relations"]) * 5
        return c
    
    def info(self):
        return {
            "type": "graph",
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "relation_types": list(set(
                r for e in self.edges for r in e["relations"]
            ))
        }


def compute_relations(n1, n2):
    """Compute spatial relations between two object nodes."""
    r1, c1, r2, c2 = n1["bbox"]
    r3, c3, r4, c4 = n2["bbox"]
    cx1, cy1 = n1["center"]
    cx2, cy2 = n2["center"]
    
    rels = []
    # Adjacency (touching or overlapping)
    if (r2 >= r3 - 1 and r1 <= r4 + 1 and c2 >= c3 - 1 and c1 <= c4 + 1):
        rels.append("adjacent")
    
    # Directional
    if cy1 < cy2 - 1:
        rels.append("left_of")
    elif cy1 > cy2 + 1:
        rels.append("right_of")
    if cx1 < cx2 - 1:
        rels.append("above")
    elif cx1 > cx2 + 1:
        rels.append("below")
    
    # Containment
    if r3 <= r1 and r4 >= r2 and c3 <= c1 and c4 >= c2:
        rels.append("contains")
    elif r1 <= r3 and r2 >= r4 and c1 <= c3 and c2 >= c4:
        rels.append("inside")
    
    # Alignment
    if abs(cy1 - cy2) < 1.5:
        rels.append("h_aligned")
    if abs(cx1 - cx2) < 1.5:
        rels.append("v_aligned")
    
    # Size comparison
    if n1["area"] > n2["area"] * 1.5:
        rels.append("larger_than")
    elif n2["area"] > n1["area"] * 1.5:
        rels.append("smaller_than")
    
    return rels


class MultiParseForest:
    """
    Layer 6: Multiple competing segmentations.
    Maintains all views and ranks by MDL score + invertibility.
    """
    
    def __init__(self, g):
        self.grid = copy_grid(g)
        self.views = {}
        self._build_views()
    
    def _build_views(self):
        for cls in [RawPixelView, ConnectedComponentView, ColorMaskView,
                    BoundingBoxView, RelationalGraphView]:
            try:
                self.views[cls.__name__] = cls.from_grid(self.grid)
            except Exception:
                pass
    
    def ranked_views(self):
        """Return views sorted by MDL cost (lower = more compact)."""
        items = [(v.mdl_cost(), name, v) for name, v in self.views.items()]
        items.sort(key=lambda x: x[0])
        return items
    
    def best_view(self):
        """Return the most compact invertible representation."""
        ranked = self.ranked_views()
        return ranked[0][2] if ranked else None
    
    def info(self):
        return {
            name: v.info() for name, v in self.views.items()
        }


if __name__ == "__main__":
    import json, glob, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Test on sample tasks
    task_files = sorted(glob.glob("/tmp/kilo/arc_data/training/*.json"))[:5]
    for tf in task_files:
        with open(tf) as f:
            task = json.load(f)
        tid = os.path.basename(tf).replace(".json", "")
        
        g_in = task["train"][0]["input"]
        g_out = task["train"][0]["output"]
        
        forest_in = MultiParseForest(g_in)
        forest_out = MultiParseForest(g_out)
        
        print(f"\n=== Task {tid} ===")
        print(f"  Input views (MDL ranking):")
        for cost, name, view in forest_in.ranked_views():
            print(f"    {name}: cost={cost}, info={view.info()}")
        print(f"  Output views:")
        for cost, name, view in forest_out.ranked_views():
            print(f"    {name}: cost={cost}")
    
    print("\nRepresentations module working!")
