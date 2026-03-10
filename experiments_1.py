import random
import make_index
import knn_search
import knn_R_tree

# Bounding box constants
X_MIN, X_MAX = -90.0, 90.0
Y_MIN, Y_MAX = -176.3, 177.5
i = 0
random.seed(42)
random_query_points = []
list_n = [10, 50, 100, 150, 200]
for _ in range(100):
    random_query_points.append((random.uniform(X_MIN, X_MAX), random.uniform(Y_MIN, Y_MAX)))

with open('knn_r_tree_output.csv', 'w') as f:
    f.write("Algo,k,n,time,NumberOfCellsVisited\n")
with open('knn_output.csv', 'w') as f:
    f.write("Algo,k,n,time,NumberOfCellsVisited\n")

for n in list_n:
    args = {'data_path': 'Gowalla_totalCheckins.txt',
            'data_path_new': 'Gowalla_totalCheckinsNew.txt',
            'index_path': 'Gowalla_totalCheckinsIndex.txt',
            'n': n,
            'k': 5}

    make_index.duplicate_elimination(args['data_path'], args['data_path_new'])
    make_index.create_index(args['data_path_new'], args['index_path'], args['n'])

    for point in random_query_points:
        if i==0:
            knn_search.knn_linear_scan(point[0], point[1], args['data_path_new'], args['k']) #since n doesn't affect the linear scan, we only run it once for the first n value
            knn_R_tree.knn_R_tree(point[0], point[1], args['data_path_new'], args['k']) #since n doesn't affect the knn search on R-tree, we only run it once for the first n value
        knn_search.knn_grid(point[0], point[1], args['index_path'], args['k'], args['n'])
        knn_search.knn_grid_bf(point[0], point[1], args['index_path'], args['k'], args['n'])
        
    i+=1
