#!/usr/bin/env python3
"""
COMP3323 Assignment 2 - Part 1: Grid Index Construction
  - Data preparation (duplicate elimination)
  - Grid index construction and file I/O
"""

import sys
import time
import argparse

# Bounding box constants
X_MIN, X_MAX = -90.0, 90.0      # latitude range
Y_MIN, Y_MAX = -176.3, 177.5    # longitude range


def duplicate_elimination(data_path, data_path_new):
    """
    Remove duplicate locations and invalid coordinates from the dataset.

    Input:
      - data_path (str):     path to original loc-gowalla_totalCheckins.txt
      - data_path_new (str): output path for cleaned dataset
    Output:
      Writes a tab-separated file: latitude\tlongitude\tlocation_id
    """
    # [YOUR CODE HERE]
    processed_dictionary = {}
    with open(data_path, 'r') as f:
        for line in f:
          lat_lan = [float(x) for x in line.split("\t")[2:4]]
          location_id = int(line.split("\t")[4])
          if (lat_lan[0] >= X_MIN and lat_lan[0] <= X_MAX) and (lat_lan[1] >= Y_MIN and lat_lan[1] <= Y_MAX):
            if (lat_lan[0], lat_lan[1]) not in processed_dictionary:
              processed_dictionary[(lat_lan[0], lat_lan[1])] = location_id
            else:
              if location_id < processed_dictionary[(lat_lan[0], lat_lan[1])]:
                processed_dictionary[(lat_lan[0], lat_lan[1])] = location_id
    f.close()
    with open(data_path_new, 'w') as f:
      for key in processed_dictionary:
        f.write(f"{key[0]}\t{key[1]}\t{processed_dictionary[key]}\n")
    f.close()

def create_index(data_path_new, index_path, n):
    """
    Build an n*n grid index from the cleaned dataset and save to disk.

    Input:
      - data_path_new (str): path to cleaned dataset
      - index_path (str):    output path for the grid index file
      - n (int):             grid size (n x n cells)
    Output:
      Writes index file with format: Cell row, col: id_lat_lon id_lat_lon ...
    """
    # [YOUR CODE HERE]
    grid = {}
    for i in range(n):
      for j in range(n):
        grid[(i, j)] = []
    with open(data_path_new, 'r') as f:
        for line in f:
          lat_lan = [float(x) for x in line.split("\t")[0:2]]
          location_id = int(line.split("\t")[2])
          col = min(int(((lat_lan[0] - X_MIN) / (X_MAX - X_MIN))*n), n-1)
          row = min(int(((lat_lan[1]-Y_MIN) / (Y_MAX - Y_MIN))*n), n-1)
          grid[(row, col)].append(f'{location_id}_{lat_lan[0]}_{lat_lan[1]}')
    f.close()
    sorted_keys = sorted(grid.keys())
    with open(index_path, 'w') as f:      
      for key in sorted_keys:
        f.write(f"Cell {int(key[0])}, {int(key[1])}: {' '.join(grid[key])}\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Part 1: Data preparation and grid index construction")
    parser.add_argument("data_path",
                        help="path to original Gowalla_totalCheckins.txt")
    parser.add_argument("index_path",
                        help="output path for the grid index file")
    parser.add_argument("data_path_new",
                        help="output path for deduplicated dataset")
    parser.add_argument("n", type=int,
                        help="grid size (n x n cells)")
    args = parser.parse_args()

    duplicate_elimination(args.data_path, args.data_path_new)

    s = time.time()
    create_index(args.data_path_new, args.index_path, args.n)
    t = time.time()
    print(f"Index construction time: {(t - s) * 1000:.2f} ms")
