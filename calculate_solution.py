import os
import random
from itertools import combinations
import numpy as np
import pandas as pd
import vrplib
from typing import *
from pathlib import Path
import time
from tqdm import tqdm
import cProfile


def move_element(base_list: List[Any], obj_idx: int, new_position: int) -> List[Any]:
    new_list = base_list.copy()
    new_list.insert(new_position, new_list.pop(obj_idx))
    return new_list


def swap_positions(base_list: List[Any], position1: int, position2: int) -> List[Any]:
    """Swaps the position of 2 list elements

    :param base_list: List whose elements will be swapped
    :param position1: First position to swap
    :param position2: Second position to swap
    :return: List with swapped elements
    """
    base_list[position1], base_list[position2] = base_list[position2], base_list[position1]
    return base_list


def calculate_route_length(route: List[int]):
    return sum([np.linalg.norm(point - route[num]) for num, point in enumerate(route[1:])])


def get_nearest_neighbour_solution(instance: Dict[str, Any], distance_matrix: np.ndarray, instance_name: str):
    points = [num for num, _ in enumerate(instance['node_coord'])][1:]
    capacity_limit = instance['capacity']
    demands = list(instance['demand'])[1:]
    routes = []
    while points:
        current_route = []
        start = np.argmin(distance_matrix[0, points])
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route += [current_point]

        while capacity < capacity_limit and points:
            next_step = np.argmin(distance_matrix[current_point, points])
            if capacity + demands[next_step] <= capacity_limit:
                current_point = points.pop(next_step)
                current_route += [current_point]
                capacity += demands.pop(next_step)
            else:
                break
        routes += [current_route]
    cost = calculate_cost(instance, {'routes': routes})
    vrplib.write_solution(f'solutions/NearestNeighbour/{instance_name}.sol', routes,
                          {'cost': cost})
    return cost


def calculate_cost(instance: dict[str, Any], solution: dict[str, Any], rounding: bool = True) -> Union[int, float]:
    cost = 0
    edges = instance['node_coord']
    for route in solution['routes']:
        last = 0
        for edge in route:
            cost += round(np.linalg.norm(edges[edge] - edges[last])) if rounding else np.linalg.norm(
                edges[edge] - edges[last])
            last = edge
        cost += round(np.linalg.norm(edges[last] - edges[0])) if rounding else np.linalg.norm(edges[last] - edges[0])
    return cost


def run_cheapest_insertion_loop(points: List[int],
                                capacity_limit: int, demands: List[int],
                                distance_matrix: np.ndarray):
    routes = []
    # route_history = []
    while points:
        # sub_hist = []
        current_route = [0]
        start = np.argmin(distance_matrix[0, points])
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route += [current_point, 0]
        distances = [distance_matrix[0, current_point], distance_matrix[0, current_point]]
        # sub_hist.append(np.array(current_route))
        while capacity < capacity_limit and points:
            insertion_prices = [distance_matrix[current_route[num], edge] + distance_matrix[
                current_route[num + 1], edge] - old_dist for edge in points for num, old_dist in
                                enumerate(distances)]
            next_step = np.argmin(np.array(insertion_prices))
            next_edge_index = next_step // len(distances)
            insertion_index = next_step % len(distances)

            if capacity + demands[next_edge_index] <= capacity_limit:
                current_point = points.pop(next_edge_index)
                current_route.insert(insertion_index + 1, current_point)
                capacity += demands.pop(next_edge_index)
                distances.insert(insertion_index, insertion_prices[next_step])
            else:
                break
        #    sub_hist.append(np.array(current_route))
        # sub_hist.append(np.array(current_route))
        routes += [current_route[1:-1]]
        # route_history.append(sub_hist)
    # route_history = [[j.tolist() for j in i] for i in route_history]
    return routes  # , route_history


def get_cheapest_insert_index(current_edge_numbers: List[int], new_edge: int, distance_matrix: np.ndarray) -> int:
    return np.argmin(
        np.array([distance_matrix[current_edge, new_edge] + distance_matrix[current_edge_numbers[num], new_edge]
                  - distance_matrix[current_edge_numbers[num], current_edge] for num, current_edge in
                  enumerate(current_edge_numbers[1:])])) + 1


def run_nearest_insertion_loop(points: List[int],
                               capacity_limit: int, demands: List[int],
                               distance_matrix: np.ndarray):
    routes = []
    # route_history = []
    while points:
        # sub_hist = []
        current_route = [0]
        start = np.argmin(distance_matrix[0, points])
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route += [current_point, 0]
        # sub_hist.append(np.array(current_route))
        while capacity < capacity_limit and points:
            next_edge_index = np.argmin(np.array(
                [np.min([distance_matrix[current_route_index, point] for current_route_index in current_route])
                 for
                 point in points]))
            if capacity + demands[next_edge_index] <= capacity_limit:
                next_edge = points.pop(next_edge_index)
                insertion_index = get_cheapest_insert_index(current_route, next_edge, distance_matrix)
                current_route.insert(insertion_index, next_edge)
                capacity += demands.pop(next_edge_index)

            else:
                break
        #    sub_hist.append(np.array(current_route))
        # sub_hist.append(np.array(current_route))
        routes += [current_route[1:-1]]
        # route_history.append(sub_hist)
    # route_history = [[j.tolist() for j in i] for i in route_history]
    return routes  # , route_history


def run_farthest_insertion_loop(points: List[int],
                                capacity_limit: int, demands: List[int],
                                distance_matrix: np.ndarray):
    routes = []
    # route_history = []
    while points:
        # sub_hist = []
        current_route = [0]

        start = np.argmax(distance_matrix[0, points])
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route += [current_point, 0]
        # sub_hist.append(np.array(current_route))
        while capacity < capacity_limit and points:
            next_edge_index = np.argmax(np.array(
                [np.max([distance_matrix[current_point_index, point] for current_point_index in current_route])
                 for
                 point in points]))
            if capacity + demands[next_edge_index] <= capacity_limit:
                next_edge = points.pop(next_edge_index)
                insertion_index = get_cheapest_insert_index(current_route, next_edge, distance_matrix)
                current_route.insert(insertion_index, next_edge)
                capacity += demands.pop(next_edge_index)

            else:
                break
            # sub_hist.append(np.array(current_route))
        # sub_hist.append(np.array(current_route))
        routes += [current_route[1:-1]]
        # route_history.append(sub_hist)
    # route_history = [[j.tolist() for j in i] for i in route_history]
    return routes  # , route_history


def run_random_insertion_loop(points: List[int],
                              capacity_limit: int, demands: List[int],
                              distance_matrix: np.ndarray):
    routes = []
    # route_history = []
    while points:
        # sub_hist = []
        current_route = [0]
        start = random.randrange(0, len(points))
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route += [current_point, 0]
        # sub_hist.append(np.array(current_route))
        while capacity < capacity_limit and points:
            next_edge_index = random.randrange(0, len(points))
            if capacity + demands[next_edge_index] <= capacity_limit:
                next_edge = points.pop(next_edge_index)
                next_edge_number = next_edge
                insertion_index = get_cheapest_insert_index(current_route, next_edge, distance_matrix)
                current_route.insert(insertion_index, next_edge_number)
                capacity += demands.pop(next_edge_index)

            else:
                break
            # sub_hist.append(np.array(current_route))
        # sub_hist.append(np.array(current_route))
        routes += [current_route]
        # route_history.append(sub_hist)
    # route_history = [[j.tolist() for j in i] for i in route_history]
    return routes  # , route_history


def get_insert_solution(instance: Dict[str, Any], distance_matrix: np.ndarray, instance_name: str,
                        solver_type: str = 'cheapest'
                        ):
    points = [num for num, _ in enumerate(instance['node_coord'])][1:]
    capacity_limit = instance['capacity']
    demands = list(instance['demand'])[1:]
    if solver_type == 'cheapest':
        routes = run_cheapest_insertion_loop(points=points, capacity_limit=capacity_limit, demands=demands,
                                             distance_matrix=distance_matrix)
    elif solver_type == 'nearest':
        routes = run_nearest_insertion_loop(points=points, capacity_limit=capacity_limit, demands=demands,
                                            distance_matrix=distance_matrix)
    elif solver_type == 'farthest':
        routes = run_farthest_insertion_loop(points=points, capacity_limit=capacity_limit, demands=demands,
                                             distance_matrix=distance_matrix)
    elif solver_type == 'random':
        routes = run_random_insertion_loop(points=points, capacity_limit=capacity_limit, demands=demands,
                                           distance_matrix=distance_matrix)

    else:
        raise ValueError(
            'The solver type you are looking for is not implemented, make sure that solver_type is one out of ["cheapest", "nearest", "farthest", "random"]')

    # with open(f'solutions/Insertion/{solver_type}/{instance_path.rsplit(r"/", 1)[1].split(".")[0]}.txt', 'w',
    #          encoding='utf-8') as f:
    #    f.write(str(route_history))
    cost = calculate_cost(instance, {'routes': routes})
    vrplib.write_solution(f'solutions/Insertion/{solver_type}/{instance_name}.sol',
                          routes,
                          {'cost': cost})
    return cost


def get_savings_solution(instance: Dict[str, Any], distance_matrix: np.ndarray, instance_name: str):
    route_history = []
    points = [num for num, _ in enumerate(instance['node_coord'])][1:]
    capacity_limit = instance['capacity']
    demands = {num + 1: demand for num, demand in enumerate(instance['demand'][1:])}
    routes = [[num + 1] for num, _ in enumerate(points)]

    savings = [(distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j],
                (i, j)) for i in points for j in points if
               i > j]
    savings = sorted(savings, key=lambda x: x[0], reverse=True)
    for saving in savings:
        if saving[0] <= 0.1: continue
        route_history += [routes]
        i, j = saving[1]
        try:
            first_route = list(filter(lambda x: x[0] == i or x[-1] == i, routes))[0]
            second_route = list(filter(lambda x: x[0] == j or x[-1] == j, routes))[0]
        except IndexError:
            continue
        if first_route == second_route or first_route == second_route[::-1]: continue
        if np.sum([demands[num] for num in first_route] + [demands[num] for num in second_route]) <= capacity_limit:
            if i == first_route[0]:
                first_route = first_route[::-1]
            if j == second_route[-1]:
                second_route = second_route[::-1]

            routes = list(
                map(lambda x: first_route + second_route if x == first_route or x == first_route[::-1] else x, routes))
            try:
                routes.remove(second_route)
            except ValueError:
                routes.remove(second_route[::-1])
    route_history += routes
    with open(f'solutions/Savings/{instance_name}.txt', 'w',
              encoding='utf-8') as f:
        f.write(str(route_history))
    cost = calculate_cost(instance, {'routes': routes})
    vrplib.write_solution(f'solutions/Savings/{instance_name}.sol',
                          routes,
                          {'cost': cost})
    return cost


def sort_by_polar_angle(points):
    def key(x):
        x = x[0]
        atan = np.arctan2(x[1], x[0])
        return atan if atan >= 0 else 2 * np.pi + atan

    return sorted(points, key=key)


def get_sweep_solution(instance: Dict[str, Any], distance_matrix: np.ndarray, instance_name: str):
    depot_location = instance['node_coord'][0]
    points = list(instance['node_coord'])[1:]
    capacity_limit = instance['capacity']
    demands = list(instance['demand'])
    point_number_map = {tuple(edge): num for num, edge in enumerate(instance['node_coord'])}

    angles = sort_by_polar_angle([(point - depot_location, num + 1) for num, point in enumerate(points)])
    routes = []
    while angles:

        route = []
        while angles and sum([demands[num] for num, _ in enumerate(route)]) + demands[angles[0][1]] <= capacity_limit:
            route += [angles.pop(0)[1]]
        routes += [route]

    for num, route in enumerate(routes):
        route = run_nearest_insertion_loop(points=route,
                                           capacity_limit=capacity_limit, demands=demands,
                                           distance_matrix=distance_matrix)
        routes[num] = route[0]
    cost = calculate_cost(instance, {'routes': routes})
    vrplib.write_solution(f'solutions/Sweep/{instance_name}.sol',
                          routes,
                          {'cost': cost})

    return cost


def get_relocated_route(route: List[np.ndarray]) -> List[np.ndarray]:
    if len(route) <= 1: return route
    possible_routes = [[calculate_route_length(move_element(route, i, j)), (i, j)] for i, _ in enumerate(route) for
                       j, _ in enumerate(route) if i < j]
    best_route = min(possible_routes, key=lambda x: x[0])
    return move_element(route, best_route[1][0], best_route[1][1])


def get_exchanged_route(route: List[np.ndarray]) -> List[np.ndarray]:
    if len(route) <= 1: return route
    possible_routes = [[calculate_route_length(swap_positions(route, i, j)), (i, j)] for i, _ in enumerate(route) for
                       j, _ in enumerate(route) if i < j]
    best_route = min(possible_routes, key=lambda x: x[0])
    return swap_positions(route, best_route[1][0], best_route[1][1])


def benchmark_opening_heuristics(benchmark_folder_path: Union[os.PathLike, str] = r'data/Seminar_Arbeit_Subset'):
    benchmark_info = {'Benchmark': [], 'Algorithmus': [], 'Lösungszeit': [], 'Routenkosten': [],
                      'Relative Differenz zur optimalen Lösung': []}
    for data_path in tqdm(list(Path(benchmark_folder_path).glob('**/*.vrp'))):
        data_path = str(data_path).replace('\\', '/')
        benchmark = vrplib.read_instance(data_path)
        benchmark_name = data_path.rsplit("/", 1)[1].split(".")[0]
        dist_mat = get_distance_matrix(benchmark['node_coord'])
        optimal_cost = vrplib.read_solution(data_path.replace('.vrp', '.sol'))['cost']
        for solver_type in ['cheapest', 'nearest', 'farthest', 'random']:
            benchmark_info['Benchmark'] += [benchmark_name]
            benchmark_info['Algorithmus'] += [f'{solver_type} Insert']
            t_0 = time.perf_counter()
            cost = get_insert_solution(instance=benchmark, solver_type=solver_type, instance_name=benchmark_name,
                                       distance_matrix=dist_mat)
            t_f = time.perf_counter()
            benchmark_info['Lösungszeit'] += [t_f - t_0]
            benchmark_info['Routenkosten'] += [cost]
            benchmark_info['Relative Differenz zur optimalen Lösung'] += [
                f'{100 * (cost - optimal_cost) / optimal_cost:.2f}%']

        for solver, solver_name in zip([get_sweep_solution, get_savings_solution, get_nearest_neighbour_solution],
                                       ['Sweep', 'Savings', 'Nearest Neighbour']):
            benchmark_info['Benchmark'] += [benchmark_name]
            benchmark_info['Algorithmus'] += [solver_name]
            t_0 = time.perf_counter()
            cost = solver(instance=benchmark, distance_matrix=dist_mat, instance_name=benchmark_name)
            t_f = time.perf_counter()
            benchmark_info['Lösungszeit'] += [t_f - t_0]
            benchmark_info['Routenkosten'] += [cost]
            benchmark_info['Relative Differenz zur optimalen Lösung'] += [
                f'{100 * (cost - optimal_cost) / optimal_cost:.2f}%']

    df = pd.DataFrame(benchmark_info)
    df.to_csv(f'Benchmark_Results/results_benchmark_{benchmark_folder_path.rsplit("/", 1)[1]}.csv')


def get_distance_matrix(points: List[np.ndarray]) -> np.ndarray:
    dist_mat = np.zeros((len(points), len(points)))
    for i in range(dist_mat.shape[0]):
        for j in range(dist_mat.shape[0]):
            dist_mat[i, j] = np.linalg.norm(points[i] - points[j])
    return dist_mat


if __name__ == '__main__':
    benchmark_opening_heuristics()
#
