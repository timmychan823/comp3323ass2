from math import sqrt, ceil
import time
from typing import List, Tuple, Optional, Any
import heapq
import argparse

Point = Tuple[float, float, int]  # (x, y, location_id)

def load_points_from_file(file_path: str) -> List[Point]:
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            lat, lon, location_id = line.strip().split('\t')
            points.append((float(lat), float(lon), int(location_id)))
    return points

def dist(x1, y1, x2, y2) -> float:
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class MBR:
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def distance_to_point(self, x: float, y: float) -> float:
        dx = 0.0
        dy = 0.0
        if x < self.xmin:
            dx = self.xmin - x
        elif x > self.xmax:
            dx = x - self.xmax
        if y < self.ymin:
            dy = self.ymin - y
        elif y > self.ymax:
            dy = y - self.ymax
        return sqrt(dx * dx + dy * dy)

class Node:
    def __init__(self, is_leaf: bool, children: List["Node"|Point]):
        self.is_leaf = is_leaf
        self.children: List["Node"|Point] = children  # Points for leaf, Node for internal
        self.bbox: MBR = Node._calc_bbox(is_leaf, children)

    @classmethod
    def _calc_bbox(cls, is_current_node_leaf: bool, children: List[Any]) -> MBR:
        if is_current_node_leaf:
            x_min = min([child[0] for child in children])
            y_min = min([child[1] for child in children])
            x_max = max([child[0] for child in children])
            y_max = max([child[1] for child in children])
        else:
            x_min = min([child.bbox.xmin for child in children])
            y_min = min([child.bbox.ymin for child in children])
            x_max = max([child.bbox.xmax for child in children])
            y_max = max([child.bbox.ymax for child in children])

        bbox = MBR(x_min, y_min, x_max, y_max)
        return bbox

class RTree:
    def __init__(self, max_entries: int = 8):
        assert max_entries >= 2
        self.max_entries = max_entries
        self.root: Optional[Node] = None

    def bulk_load(self, points: List[Point]):
        if not points:
            self.root = Node(is_leaf=True)
            return

        M = self.max_entries
        n = len(points)

        # sort points by x, partition into S slices
        points_sorted_x = sorted(points, key=lambda p: p[0])
        leaf_nodes = []
        for i in range(0, n, M):
            if i+M > n:
                # last slice may have fewer than M points, but we can still pack it as a leaf node
                leaf_nodes.append(Node(is_leaf=True, children=points_sorted_x[i:]))
            else:
                leaf_nodes.append(Node(is_leaf=True, children=points_sorted_x[i:i + M]))

        # build upper layers
        current_level = leaf_nodes
        while len(current_level) > 1:
            # sort nodes by their bbox's xmin to get spatial locality
            current_level.sort(key=lambda node: node.bbox.xmin)
            n = len(current_level)
            non_leaf_nodes = []
            for i in range(0, n, M):
                if i+M >= n:
                    # last slice may have fewer than M points, but we can still pack it as a leaf node            slice_block = points_sorted_x[i:i + M]
                    non_leaf_nodes.append(Node(is_leaf=False, children=current_level[i:]))
                else:
                    non_leaf_nodes.append(Node(is_leaf=False, children=current_level[i:i + M]))
            next_level = non_leaf_nodes
            current_level = next_level
        self.root = current_level[0] if current_level else Node(is_leaf=True)


    #using best first search to find the k nearest neighbors of the query point (qx, qy) in the R-tree
    def knn_R_tree(self, qx: float, qy: float, k: int = 1) -> tuple[str, int]:
        s = time.time()
        knn_max_heap = [[float('-inf'),(float('-inf'),float('-inf'),-1)] for _ in range(k)]
        heapq.heapify(knn_max_heap)
        min_heap = [[child.bbox.distance_to_point(qx, qy), child] for child in self.root.children]
        
        while len(min_heap) != 0 and min_heap[0][0] < -knn_max_heap[0][0]:
            e = heapq.heappop(min_heap)
            if not e[1].is_leaf:
                for node in e[1].children:
                    dist_node_q = node.bbox.distance_to_point(qx, qy)
                    if dist_node_q < -knn_max_heap[0][0]:
                        heapq.heappush(min_heap, [dist_node_q, node])
            else:
                for entry in e[1].children:
                    dist_entry_q = dist(qx, qy, entry[0], entry[1])
                    if dist_entry_q < -knn_max_heap[0][0]:
                        heapq.heappush(knn_max_heap, [-dist_entry_q, entry])
                        if len(knn_max_heap) > k:
                            heapq.heappop(knn_max_heap)
        result_str = [int(heapq.heappop(knn_max_heap)[1][2]) for _ in range(len(knn_max_heap))]
        result_str.reverse()
        result_str = [str(location_id) for location_id in result_str if location_id != -1]

        t = time.time()
        
        #####for reporting results#####
        # with open('knn_r_tree_output.csv', 'a') as f:
        #     f.write(f"knn_R_tree,{k},{None},{(t - s) * 1000:.2f},{None}\n")
        #####for reporting results#####

        return ", ".join(result_str), 0  # number of cells visited is not applicable for R-tree, so we return 0

def knn_R_tree(x, y, data_path_new, k):
    pts: List[Point] = load_points_from_file(data_path_new)
    r_tree = RTree(max_entries=3)
    r_tree.bulk_load(pts)
    return r_tree.knn_R_tree(x, y, k=k)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Parts 5: Run k-NN search using knn_R_tree")
    parser.add_argument("x", type=float,
                        help="latitude of the query point q")
    parser.add_argument("y", type=float,
                        help="longitude of the query point q")
    parser.add_argument("data_path_new",
                        help="path to the deduplicated dataset")
    parser.add_argument("k", type=int,
                        help="number of nearest neighbors")
    args = parser.parse_args()
    result, _ = knn_R_tree(args.x, args.y, args.data_path_new, args.k)
    print(f"knn_R_tree results: {result}")