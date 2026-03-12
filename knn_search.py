#!/usr/bin/env python3
"""
COMP3323 Assignment 2 - Parts 2, 3 & 4(1): k-NN Search Algorithms
  - knn_linear_scan  (Part 4.1)
  - knn_grid         (Part 2: layer-by-layer expansion)
  - knn_grid_bf      (Part 3: best-first cell expansion)
"""
from math import sqrt, ceil
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

def dlow(n, grid_index, cell, x, y): #TODO: this should not be calculated like this, we should take into the account all the points in the cell before calculating euclidean distance, should use a MBR within the cell, grid_index should also be provided
  #c should be in the format of (row, col)
  
  x_min = grid_index[cell][0][0]
  x_max = grid_index[cell][0][1]
  y_min = grid_index[cell][0][2]
  y_max = grid_index[cell][0][3]

  dx = 0.0
  dy = 0.0
  if x < x_min:
      dx = x_min - x
  elif x > x_max:
      dx = x - x_max
  if y < y_min:
      dy = y_min - y
  elif y > y_max:
      dy = y - y_max
  return sqrt(dx * dx + dy * dy)


def load_grid_index(index_path, n):
  grid_index = {}
  with open(index_path, 'r') as f:
    for line in f:
      cell_info, points_str = line.strip().split(':')
      cell_row, cell_col = map(int, cell_info.replace('Cell', '').split(','))
      points = [[int(location[0]), float(location[1]), float(location[2])] for location in [string_location.split('_') for string_location in points_str.strip().split()]]
      
      x_min = float('inf')
      y_min = float('inf')
      x_max = float('-inf')
      y_max = float('-inf')
      if len(points) == 0:
        x_min, x_max, y_min, y_max = X_MIN + cell_col*((X_MAX - X_MIN)/n), X_MIN + (cell_col+1)*((X_MAX - X_MIN)/n), Y_MIN + cell_row*((Y_MAX - Y_MIN)/n), Y_MIN + (cell_row+1)*((Y_MAX - Y_MIN)/n)
      else:
        for point in points:
          x_min = min(x_min, point[1])
          y_min = min(y_min, point[2])
          x_max = max(x_max, point[1])
          y_max = max(y_max, point[2])
      grid_index[(cell_row, cell_col)] = ((x_min, x_max, y_min, y_max), points) #TODO: should be ((x_min, x_max, y_min, y_max), points) where (x_min, x_max, y_min, y_max) is the MBR for the cell
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
    knn_max_heap = [(float('-inf'),-1) for _ in range(k)]
    heapq.heapify(knn_max_heap)
    grid_index = load_grid_index(index_path, n)
    #TODO: should add MBR for each cell in the grid index

    s = time.time()

    q_cell = {'col': int((x-X_MIN)*n/(X_MAX-X_MIN)), 'row':int((y-Y_MIN)*n/(Y_MAX-Y_MIN))}

    while (q_cell['row'] - i >= 0 or q_cell['row'] + i < n):
      is_closer_cell_found = False
      for j in range(max(q_cell['row'] - i, 0), min(q_cell['row'] + i+1, n)):
        if (j == q_cell['row'] - i or j== q_cell['row'] + i):
          for col in range(max(q_cell['col'] - i, 0), min(q_cell['col'] + i+1, n)):
            if (j, col) in grid_index:

              if dlow(n, grid_index,(j, col), x, y) >= -knn_max_heap[0][0]: #TODO: check if this should be >= or >
                continue
              
              number_of_cells_visited += 1
              is_closer_cell_found = True

              for location in grid_index[(j, col)][1]:
                distance = cal_euclidean_distance(x, y, location[1], location[2])
                if distance < -knn_max_heap[0][0]:
                    heapq.heappop(knn_max_heap)
                    heapq.heappush(knn_max_heap, (-distance, location[0]))
        else:
          col_to_scan = []
          if (q_cell['col'] - i >= 0):
            col_to_scan.append(q_cell['col'] - i)
          if (q_cell['col'] + i < n):
            col_to_scan.append(q_cell['col'] + i)
          for col in col_to_scan:
            if (j, col) in grid_index:
              if dlow(n, grid_index,(j, col), x, y) >= -knn_max_heap[0][0]: #TODO: check if this should be >= or >
                continue

              number_of_cells_visited += 1
              is_closer_cell_found = True

              for location in grid_index[(j, col)][1]:
                distance = cal_euclidean_distance(x, y, location[1], location[2])
                if distance < -knn_max_heap[0][0]:
                    heapq.heappop(knn_max_heap)
                    heapq.heappush(knn_max_heap, (-distance, location[0]))

      if is_closer_cell_found == False: 
        break
      i+=1
    
    result_str = [int(heapq.heappop(knn_max_heap)[1]) for _ in range(len(knn_max_heap))]
    result_str.reverse()
    result_str = [str(location_id) for location_id in result_str if location_id != -1]

    t = time.time()
    
    #####for reporting results#####
    # with open('knn_output.csv', 'a') as f:
    #   f.write(f"grid,{k},{n},{(t - s) * 1000:.2f},{number_of_cells_visited}\n")
    #####for reporting results#####

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
    knn_max_heap = [(float('-inf'),-1) for _ in range(k)]
    heapq.heapify(knn_max_heap)
    grid_index = load_grid_index(index_path, n)

    s = time.time()

    for i in range(n):
      for j in range(n):
        dlow_min_heap.append((dlow(n, grid_index,(i, j), x, y), i, j))
    heapq.heapify(dlow_min_heap)
  
    while len(dlow_min_heap) > 0:
      next_nearest_cell = heapq.heappop(dlow_min_heap)
      if next_nearest_cell[0] >= -knn_max_heap[0][0]: #TODO: check if this should be >= or >
        break
      number_of_cells_visited += 1
      
      if (next_nearest_cell[1], next_nearest_cell[2]) in grid_index:
        for location in grid_index[(next_nearest_cell[1], next_nearest_cell[2])][1]:
          distance = cal_euclidean_distance(x, y, location[1], location[2])
          if distance < -knn_max_heap[0][0]:
              heapq.heappop(knn_max_heap)
              heapq.heappush(knn_max_heap, (-distance, location[0]))

    result_str = [int(heapq.heappop(knn_max_heap)[1]) for _ in range(len(knn_max_heap))]
    result_str.reverse()
    result_str = [str(location_id) for location_id in result_str if location_id != -1]

    t = time.time()

    #####for reporting results#####
    # with open('knn_output.csv', 'a') as f:
    #   f.write(f"grid_bf,{k},{n},{(t - s) * 1000:.2f},{number_of_cells_visited}\n")
    #####for reporting results#####

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

    knn_max_heap = [(float('-inf'),-1) for _ in range(k)]
    heapq.heapify(knn_max_heap)

    for lat, lon, location_id in data:
      distance = cal_euclidean_distance(x, y, lat, lon)
      if distance < -knn_max_heap[0][0]:
          heapq.heappop(knn_max_heap)
          heapq.heappush(knn_max_heap, (-distance, location_id))
    result_str = [int(heapq.heappop(knn_max_heap)[1]) for _ in range(len(knn_max_heap))]
    result_str.reverse()
    result_str = [str(location_id) for location_id in result_str if location_id != -1]

    t = time.time()

    #####for reporting results#####
    # with open('knn_output.csv', 'a') as f:
    #   f.write(f"linear,{k},{None},{(t - s) * 1000:.2f},{None}\n")
    #####for reporting results#####

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
    result, _ = knn_linear_scan(args.x, args.y, args.data_path_new, args.k)
    print(f"Linear scan results: {result}")

    # Grid (layer-by-layer)
    result, cells = knn_grid(args.x, args.y, args.index_path, args.k, args.n)
    print(f"Grid (layer-by-layer) results: {result}")

    # Grid (best-first)
    result, cells = knn_grid_bf(args.x, args.y, args.index_path, args.k, args.n)
    print(f"Grid (best-first) results: {result}")