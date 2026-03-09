#!/usr/bin/env python3
"""
COMP3323 Assignment 2 - Parts 2, 3 & 4(1): k-NN Search Algorithms
  - knn_linear_scan  (Part 4.1)
  - knn_grid         (Part 2: layer-by-layer expansion)
  - knn_grid_bf      (Part 3: best-first cell expansion)
"""

import sys
import time
import math
import heapq
import argparse

# Bounding box constants
X_MIN, X_MAX = -90.0, 90.0
Y_MIN, Y_MAX = -176.3, 177.5


# ---------------------------------------------------------------------------
# You may add your own helper functions here if needed.
# ---------------------------------------------------------------------------
def cal_euclidean_distance(x1, y1, x2, y2):
  return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def dlow(n, cell, x, y):
  #c should be in the format of (row, col)
  x_min = X_MIN + cell[0] * (X_MAX - X_MIN) / n
  x_max = X_MIN + (cell[0] + 1) * (X_MAX - X_MIN) / n
  y_min = Y_MIN + cell[1] * (Y_MAX - Y_MIN) / n
  y_max = Y_MIN + (cell[1] + 1) * (Y_MAX - Y_MIN) / n

  return min(cal_euclidean_distance(x, y, x_min, y_min),
             cal_euclidean_distance(x, y, x_min, y_max),
             cal_euclidean_distance(x, y, x_max, y_min),
             cal_euclidean_distance(x, y, x_max, y_max))

def load_grid_index(index_path):
  grid_index = {}
  with open(index_path, 'r') as f:
    for line in f:
      cell_info, points_str = line.strip().split(':')
      cell_row, cell_col = map(int, cell_info.replace('Cell', '').split(','))
      points = [[int(location[0]), float(location[1]), float(location[2])] for location in [string_location.split('_') for string_location in points_str.strip().split()]]
      grid_index[(cell_row, cell_col)] = points
  return grid_index

def load_deduplicated_data(data_path_new):
  data = []
  with open(data_path_new, 'r') as f:
    for line in f:
      lat_lan = [float(x) for x in line.split("\t")[0:2]]
      location_id = int(line.split("\t")[2])
      data.append((lat_lan[0], lat_lan[1], location_id))
  return data

# ---------------------------------------------------------------------------
# Part 2: Grid k-NN (layer-by-layer expansion)
# ---------------------------------------------------------------------------

def knn_grid(x, y, index_path, k, n):
    """
    Find k nearest neighbors using grid index with layer-by-layer expansion.

    Input:
      - x (float):          latitude of query point
      - y (float):          longitude of query point
      - index_path (str):   path to the grid index file
      - k (int):            number of nearest neighbors
      - n (int):            grid size (n x n)
    Output:
      - tuple: (result_str, cells_visited)
        - result_str (str): comma-separated location ids, e.g. "11, 789, 125, 2, 771"
        - cells_visited (int): number of cells whose points were examined
    """
    # [YOUR CODE HERE]
    i = 0
    number_of_cells_visited = 0
    result_str = []
    knn_max_heap = []
    heapq.heapify(knn_max_heap)
    grid_index = load_grid_index(index_path)

    s = time.time()

    q_cell = {'row': int((x-X_MIN)*n/(X_MAX-X_MIN)), 'col':int((y-Y_MIN)*n/(Y_MAX-Y_MIN))}

    while (q_cell['row'] - i >= 0 or q_cell['row'] + i < n):
      updated = False
      for j in range(max(q_cell['row'] - i, 0), min(q_cell['row'] + i+1, n)):
        if (j == q_cell['row'] - i or j== q_cell['row'] + i):
          for col in range(max(q_cell['col'] - i, 0), min(q_cell['col'] + i+1, n)):
              if (j, col) in grid_index:
                if len(knn_max_heap) ==k and dlow(n, (j, col), x, y) > -knn_max_heap[0][0]:
                  continue
                
                number_of_cells_visited += 1 #TODO: check if this should be counted as visited for accessing dlow

                for location in grid_index[(j, col)]:
                  distance = cal_euclidean_distance(x, y, location[1], location[2])
                  if len(knn_max_heap) < k:
                    heapq.heappush(knn_max_heap, (-distance, location[0]))
                    updated = True
                  else:
                    if distance < -knn_max_heap[0][0]:
                      heapq.heappop(knn_max_heap)
                      heapq.heappush(knn_max_heap, (-distance, location[0]))
                      updated = True
        else:
          col_to_scan = []
          if (q_cell['col'] - i >= 0):
            col_to_scan.append(q_cell['col'] - i)
          if (q_cell['col'] + i < n):
            col_to_scan.append(q_cell['col'] + i)
          for col in col_to_scan:
            if (j, col) in grid_index:
              if len(knn_max_heap) !=0 and dlow(n, (j, col), x, y) > -knn_max_heap[0][0]:
                continue

              number_of_cells_visited += 1 #TODO: check if this should be counted as visited for accessing dlow

              for location in grid_index[(j, col)]:
                distance = cal_euclidean_distance(x, y, location[1], location[2])
                if len(knn_max_heap) < k:
                  heapq.heappush(knn_max_heap, (-distance, location[0]))
                  updated = True
                else:
                  if distance < -knn_max_heap[0][0]:
                    heapq.heappop(knn_max_heap)
                    heapq.heappush(knn_max_heap, (-distance, location[0]))
                    updated = True
      if updated == False and len(knn_max_heap)==k: 
        break
      i+=1
    
    result_str = [int(heapq.heappop(knn_max_heap)[1]) for _ in range(len(knn_max_heap))]
    result_str.reverse()
    result_str = [str(location_id) for location_id in result_str]

    t = time.time()
    
    with open('knn_output.csv', 'a') as f:
      f.write(f"grid,{k},{n},{(t - s) * 1000:.2f},{number_of_cells_visited}\n")

    return ", ".join(result_str), number_of_cells_visited


# ---------------------------------------------------------------------------
# Part 3: Grid k-NN (best-first cell expansion)
# ---------------------------------------------------------------------------

def knn_grid_bf(x, y, index_path, k, n):
    """
    Find k nearest neighbors using grid index with best-first cell expansion.

    Input:
      - x (float):          latitude of query point
      - y (float):          longitude of query point
      - index_path (str):   path to the grid index file
      - k (int):            number of nearest neighbors
      - n (int):            grid size (n x n)
    Output:
      - tuple: (result_str, cells_visited)
        - result_str (str): comma-separated location ids, e.g. "11, 789, 125, 2, 771"
        - cells_visited (int): number of cells whose points were examined
    """
    # [YOUR CODE HERE]
    number_of_cells_visited = 0
    result_str = []
    dlow_min_heap = []
    knn_max_heap = []
    heapq.heapify(dlow_min_heap)
    heapq.heapify(knn_max_heap)
    grid_index = load_grid_index(index_path)

    s = time.time()

    for i in range(n):
      for j in range(n):
        heapq.heappush(dlow_min_heap, (dlow(n, (i, j), x, y), i, j))
  
    while len(dlow_min_heap) > 0:
      next_nearest_cell = heapq.heappop(dlow_min_heap)
      if len(knn_max_heap) == k and next_nearest_cell[0] > -knn_max_heap[0][0]:
        break
      number_of_cells_visited += 1
      
      if (next_nearest_cell[1], next_nearest_cell[2]) in grid_index:
        for location in grid_index[(next_nearest_cell[1], next_nearest_cell[2])]:
          distance = cal_euclidean_distance(x, y, location[1], location[2])
          if len(knn_max_heap) < k:
            heapq.heappush(knn_max_heap, (-distance, location[0]))
          else:
            if distance < -knn_max_heap[0][0]:
              heapq.heappop(knn_max_heap)
              heapq.heappush(knn_max_heap, (-distance, location[0]))

    result_str = [int(heapq.heappop(knn_max_heap)[1]) for _ in range(len(knn_max_heap))]
    result_str.reverse()
    result_str = [str(location_id) for location_id in result_str]

    t = time.time()

    with open('knn_output.csv', 'a') as f:
      f.write(f"grid_bf,{k},{n},{(t - s) * 1000:.2f},{number_of_cells_visited}\n")

    return ", ".join(result_str), number_of_cells_visited

# ---------------------------------------------------------------------------
# Part 4(1): Linear scan
# ---------------------------------------------------------------------------

def knn_linear_scan(x, y, data_path_new, k):
    """
    Find k nearest neighbors by scanning all points.

    Input:
      - x (float):            latitude of query point
      - y (float):            longitude of query point
      - data_path_new (str):  path to the deduplicated dataset
      - k (int):              number of nearest neighbors
    Output:
      - tuple: (result_str, cells_visited)
        - result_str (str): comma-separated location ids, e.g. "11, 789, 125, 2, 771"
        - cells_visited (int): 0 (linear scan does not use grid cells)
    """
    # [YOUR CODE HERE]
    data = load_deduplicated_data(data_path_new)

    s = time.time()

    knn_max_heap = []
    for lat, lon, location_id in data:
      distance = cal_euclidean_distance(x, y, lat, lon)
      if len(knn_max_heap) < k:
        heapq.heappush(knn_max_heap, (-distance, location_id))
      else:
        if distance < -knn_max_heap[0][0]:
          heapq.heappop(knn_max_heap)
          heapq.heappush(knn_max_heap, (-distance, location_id))
    result_str = [int(heapq.heappop(knn_max_heap)[1]) for _ in range(len(knn_max_heap))]
    result_str.reverse()
    result_str = [str(location_id) for location_id in result_str]

    t = time.time()

    with open('knn_output.csv', 'a') as f:
      f.write(f"linear,{k},{None},{(t - s) * 1000:.2f},{None}\n")

    return ", ".join(result_str), 0
  
  
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parts 2 & 3: Run k-NN search using linear scan, "
                    "grid (layer-by-layer), and grid (best-first)")
    parser.add_argument("x", type=float,
                        help="latitude of the query point q")
    parser.add_argument("y", type=float,
                        help="longitude of the query point q")
    parser.add_argument("data_path_new",
                        help="path to the deduplicated dataset")
    parser.add_argument("index_path",
                        help="path to the grid index file")
    parser.add_argument("k", type=int,
                        help="number of nearest neighbors")
    parser.add_argument("n", type=int,
                        help="grid size (n x n cells)")
    args = parser.parse_args()

    # Linear scan
    s = time.time()
    result, _ = knn_linear_scan(args.x, args.y, args.data_path_new, args.k)
    t = time.time()
    print(f"Linear scan results: {result}")
    print(f"Linear scan time: {(t - s) * 1000:.2f} ms")

    # Grid (layer-by-layer)
    s = time.time()
    result, cells = knn_grid(args.x, args.y, args.index_path, args.k, args.n)
    t = time.time()
    print(f"Grid (layer-by-layer) results: {result}")
    print(f"Grid (layer-by-layer) time: {(t - s) * 1000:.2f} ms, cells visited: {cells}")

    # Grid (best-first)
    s = time.time()
    result, cells = knn_grid_bf(args.x, args.y, args.index_path, args.k, args.n)
    t = time.time()
    print(f"Grid (best-first) results: {result}")
    print(f"Grid (best-first) time: {(t - s) * 1000:.2f} ms, cells visited: {cells}")
