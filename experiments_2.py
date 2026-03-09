import random
import make_index
import knn_search
import knn_R_tree

# Bounding box constants
X_MIN, X_MAX = -90.0, 90.0
Y_MIN, Y_MAX = -176.3, 177.5
random.seed(42)
random_query_points = []
list_k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for _ in range(100):
    random_query_points.append((random.uniform(X_MIN, X_MAX), random.uniform(Y_MIN, Y_MAX)))

for k in list_k:
    args = {'data_path': 'Gowalla_totalCheckins.txt',
            'data_path_new': 'Gowalla_totalCheckinsNew.txt',
            'index_path': 'Gowalla_totalCheckinsIndex.txt',
            'n': 50,
            'k': k}

    make_index.duplicate_elimination(args['data_path'], args['data_path_new'])
    make_index.create_index(args['data_path_new'], args['index_path'], args['n'])

    for point in random_query_points:
        knn_search.knn_linear_scan(point[0], point[1], args['data_path_new'], args['k'])
        knn_search.knn_grid(point[0], point[1], args['index_path'], args['k'], args['n'])
        knn_search.knn_grid_bf(point[0], point[1], args['index_path'], args['k'], args['n'])
        knn_R_tree.knn_R_tree(point[0], point[1], args['data_path_new'], args['k'])


