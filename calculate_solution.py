import os
import random
from itertools import combinations, product, permutations
import numpy as np
import pandas as pd
import vrplib
from typing import *
from pathlib import Path
import time
from tqdm import tqdm


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


def get_nearest_neighbour_solution(instance_path: Union[str, os.PathLike]):
    instance = vrplib.read_instance(instance_path)
    point_map = {num: edge for num, edge in enumerate(instance['node_coord'])}
    point_number_map = {tuple(edge): num for num, edge in enumerate(instance['node_coord'])}
    points = list(instance['node_coord'])[1:]
    capacity_limit = instance['capacity']
    depot_location = point_map[0]
    demands = list(instance['demand'])[1:]
    routes = []
    while points:
        current_route = []
        start = np.argmin(np.linalg.norm(np.array(points) - depot_location, axis=1))
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route += [point_number_map[tuple(current_point)]]

        while capacity < capacity_limit and points:
            next_step = np.argmin(np.linalg.norm(np.array(points) - current_point, axis=1))
            if capacity + demands[next_step] <= capacity_limit:
                current_point = points.pop(next_step)
                current_route += [point_number_map[tuple(current_point)]]
                capacity += demands.pop(next_step)
            else:
                break
        routes += [current_route]
    cost = calculate_cost(instance, {'routes': routes})
    vrplib.write_solution(f'solutions/NearestNeighbour/{instance_path.rsplit("/", 1)[1].split(".")[0]}.sol', routes,
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


def run_cheapest_insertion_loop(point_number_map, points, capacity_limit, depot_location, demands):
    routes = []
    route_history = []
    while points:
        sub_hist = []
        current_route = []
        current_route_edges = [depot_location]
        start = np.argmin(np.linalg.norm(np.array(points) - depot_location, axis=1))
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route_edges += [current_point, depot_location]
        current_route += [point_number_map[tuple(current_point)]]
        distances = [np.linalg.norm(depot_location - current_point), np.linalg.norm(depot_location - current_point)]
        sub_hist.append(np.array(current_route))
        while capacity < capacity_limit and points:
            insertion_prices = [np.linalg.norm(current_route_edges[num] - edge) + np.linalg.norm(
                current_route_edges[num + 1] - edge) - old_dist for edge in points for num, old_dist in
                                enumerate(distances)]
            next_step = np.argmin(np.array(insertion_prices))
            next_edge_index = next_step // len(distances)
            insertion_index = next_step % len(distances)

            if capacity + demands[next_edge_index] <= capacity_limit:
                current_point = points.pop(next_edge_index)
                current_route_edges.insert(insertion_index + 1, current_point)
                current_route.insert(insertion_index + 1, point_number_map[tuple(current_point)])
                capacity += demands.pop(next_edge_index)
                distances.insert(insertion_index, insertion_prices[next_step])
            else:
                break
            sub_hist.append(np.array(current_route))
        sub_hist.append(np.array(current_route))
        routes += [current_route]
        route_history.append(sub_hist)
    route_history = [[j.tolist() for j in i] for i in route_history]
    return routes, route_history


def get_cheapest_insert_index(current_edges: List[np.ndarray], new_edge: np.ndarray) -> int:
    return np.argmin(np.array([np.linalg.norm(new_edge - current_edges[num]) + np.linalg.norm(
        current_edge - new_edge) - np.linalg.norm(current_edge - current_edges[num]) for num, current_edge in
                               enumerate(current_edges[1:])])) + 1


def run_nearest_insertion_loop(point_number_map, points, capacity_limit, depot_location, demands):
    routes = []
    route_history = []
    while points:
        sub_hist = []
        current_route = []
        current_route_edges = [depot_location]
        start = np.argmin(np.linalg.norm(np.array(points) - depot_location, axis=1))
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route_edges += [current_point, depot_location]
        current_route += [point_number_map[tuple(current_point)]]
        sub_hist.append(np.array(current_route))
        while capacity < capacity_limit and points:
            next_edge_index = np.argmin(np.array(
                [min([np.linalg.norm(current_route_point - point) for current_route_point in current_route_edges]) for
                 point in points]))
            if capacity + demands[next_edge_index] <= capacity_limit:
                next_edge = points.pop(next_edge_index)
                next_edge_number = point_number_map[tuple(next_edge)]
                insertion_index = get_cheapest_insert_index(current_route_edges, next_edge)

                current_route_edges.insert(insertion_index, next_edge)
                current_route.insert(insertion_index, next_edge_number)

                capacity += demands.pop(next_edge_index)

            else:
                break
            sub_hist.append(np.array(current_route))
        sub_hist.append(np.array(current_route))
        routes += [current_route]
        route_history.append(sub_hist)
    route_history = [[j.tolist() for j in i] for i in route_history]
    return routes, route_history


def run_farthest_insertion_loop(point_number_map, points, capacity_limit, depot_location, demands):
    routes = []
    route_history = []
    while points:
        sub_hist = []
        current_route = []
        current_route_edges = [depot_location]
        start = np.argmax(np.linalg.norm(np.array(points) - depot_location, axis=1))
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route_edges += [current_point, depot_location]
        current_route += [point_number_map[tuple(current_point)]]
        sub_hist.append(np.array(current_route))
        while capacity < capacity_limit and points:
            next_edge_index = np.argmax(np.array(
                [max([np.linalg.norm(current_route_point - point) for current_route_point in current_route_edges]) for
                 point in points]))
            if capacity + demands[next_edge_index] <= capacity_limit:
                next_edge = points.pop(next_edge_index)
                next_edge_number = point_number_map[tuple(next_edge)]
                insertion_index = get_cheapest_insert_index(current_route_edges, next_edge)
                current_route_edges.insert(insertion_index, next_edge)
                current_route.insert(insertion_index, next_edge_number)
                capacity += demands.pop(next_edge_index)

            else:
                break
            sub_hist.append(np.array(current_route))
        sub_hist.append(np.array(current_route))
        routes += [current_route]
        route_history.append(sub_hist)
    route_history = [[j.tolist() for j in i] for i in route_history]
    return routes, route_history


def run_random_insertion_loop(point_number_map, points, capacity_limit, depot_location, demands):
    routes = []
    route_history = []
    while points:
        sub_hist = []
        current_route = []
        current_route_edges = [depot_location]
        start = random.randrange(0, len(points))
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route_edges += [current_point, depot_location]
        current_route += [point_number_map[tuple(current_point)]]
        sub_hist.append(np.array(current_route))
        while capacity < capacity_limit and points:
            next_edge_index = random.randrange(0, len(points))
            if capacity + demands[next_edge_index] <= capacity_limit:
                next_edge = points.pop(next_edge_index)
                next_edge_number = point_number_map[tuple(next_edge)]
                insertion_index = get_cheapest_insert_index(current_route_edges, next_edge)
                current_route_edges.insert(insertion_index, next_edge)
                current_route.insert(insertion_index, next_edge_number)
                capacity += demands.pop(next_edge_index)

            else:
                break
            sub_hist.append(np.array(current_route))
        sub_hist.append(np.array(current_route))
        routes += [current_route]
        route_history.append(sub_hist)
    route_history = [[j.tolist() for j in i] for i in route_history]
    return routes, route_history


def get_insert_solution(instance_path: Union[os.PathLike, str], solver_type: str = 'cheapest'):
    instance = vrplib.read_instance(instance_path)
    point_map = {num: edge for num, edge in enumerate(instance['node_coord'])}
    point_number_map = {tuple(edge): num for num, edge in enumerate(instance['node_coord'])}
    points = list(instance['node_coord'])[1:]
    capacity_limit = instance['capacity']
    depot_location = point_map[0]
    demands = list(instance['demand'])[1:]
    if solver_type == 'cheapest':
        routes, route_history = run_cheapest_insertion_loop(point_number_map=point_number_map, points=points,
                                                            capacity_limit=capacity_limit, demands=demands,
                                                            depot_location=depot_location)
    elif solver_type == 'nearest':
        routes, route_history = run_nearest_insertion_loop(point_number_map=point_number_map, points=points,
                                                           capacity_limit=capacity_limit, demands=demands,
                                                           depot_location=depot_location)
    elif solver_type == 'farthest':
        routes, route_history = run_farthest_insertion_loop(point_number_map=point_number_map, points=points,
                                                            capacity_limit=capacity_limit, demands=demands,
                                                            depot_location=depot_location)
    elif solver_type == 'random':
        routes, route_history = run_random_insertion_loop(point_number_map=point_number_map, points=points,
                                                          capacity_limit=capacity_limit, demands=demands,
                                                          depot_location=depot_location)

    else:
        raise ValueError(
            'The solver type you are looking for is not implemented, make sure that solver_type is one out of ["cheapest", "nearest", "farthest", "random"]')

    with open(f'solutions/Insertion/{solver_type}/{instance_path.rsplit(r"/", 1)[1].split(".")[0]}.txt', 'w',
              encoding='utf-8') as f:
        f.write(str(route_history))
    cost = calculate_cost(instance, {'routes': routes})
    vrplib.write_solution(f'solutions/Insertion/{solver_type}/{instance_path.rsplit("/", 1)[1].split(".")[0]}.sol',
                          routes,
                          {'cost': cost})
    return cost


def get_savings_solution(instance_path: Union[str, os.PathLike]):
    instance = vrplib.read_instance(instance_path)
    route_history = []
    depot_location = instance['node_coord'][0]
    points = list(instance['node_coord'])[1:]
    capacity_limit = instance['capacity']
    demands = {num + 1: demand for num, demand in enumerate(instance['demand'][1:])}
    routes = [[num + 1] for num, _ in enumerate(points)]

    savings = [(np.linalg.norm(depot_location - i) + np.linalg.norm(depot_location - j) - np.linalg.norm(i - j),
                (num_i + 1, num_j + 1)) for num_i, i in enumerate(points) for num_j, j in
               enumerate(points) if
               num_i >= num_j and not np.array_equal(i, j)]
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
        if sum([demands[num] for num in first_route] + [demands[num] for num in second_route]) <= capacity_limit:
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
    with open(f'solutions/Savings/{instance_path.rsplit("/", 1)[1].split(".")[0]}.txt', 'w',
              encoding='utf-8') as f:
        f.write(str(route_history))
    cost = calculate_cost(instance, {'routes': routes})
    vrplib.write_solution(f'solutions/Savings/{instance_path.rsplit("/", 1)[1].split(".")[0]}.sol',
                          routes,
                          {'cost': cost})
    return cost

def sort_by_polar_angle(points):
   def key(x):
        x = x[0]
        atan = np.arctan2(x[1], x[0])
        return atan if atan >= 0 else 2*np.pi + atan
   return sorted(points, key=key)


def get_sweep_solution(instance_path: Union[str, os.PathLike]):
    instance = vrplib.read_instance(instance_path)
    depot_location = instance['node_coord'][0]
    point_map = {num: edge for num, edge in enumerate(instance['node_coord'])}
    points = list(instance['node_coord'])[1:]
    capacity_limit = instance['capacity']
    demands = {num: demand for num, demand in enumerate(instance['demand'])}
    point_number_map = {tuple(edge): num for num, edge in enumerate(instance['node_coord'])}

    angles = sort_by_polar_angle([(point-depot_location, num + 1) for num, point in enumerate(points)])
    routes = []
    while angles:

        route = []
        while angles and sum([demands[knot] for knot in route]) + demands[angles[0][1]] <= capacity_limit:
            route += [angles.pop(0)[1]]
        routes += [route]



    for num, route in enumerate(routes):
        route, _ = run_nearest_insertion_loop(point_number_map, [point_map[point_number] for point_number in route],
                                              capacity_limit,depot_location,[0 for _ in route])
        routes[num] = route[0]
    cost = calculate_cost(instance, {'routes': routes})
    vrplib.write_solution(f'solutions/Sweep/{instance_path.rsplit("/", 1)[1].split(".")[0]}.sol',
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


def get_lambda_opt(route: List[np.ndarray], _lambda: int = 2) -> List[np.ndarray]:
    possible_sub_routes = list(combinations(route, len(route) - _lambda))


def benchmark_opening_heuristics(benchmark_folder_path: Union[os.PathLike, str] = r'data/Seminar_Arbeit_Subset'):
    benchmark_info = {'Benchmark': [], 'Algorithmus': [], 'Lösungszeit': [], 'Routenkosten': [],
                      'Relative Differenz zur optimalen Lösung': []}
    for data_path in tqdm(list(Path(benchmark_folder_path).glob('**/*.vrp'))):

        data_path = str(data_path).replace('\\', '/')
        optimal_cost = vrplib.read_solution(data_path.replace('.vrp', '.sol'))['cost']
        for solver_type in ['cheapest', 'nearest', 'farthest', 'random']:
            benchmark_info['Benchmark'] += [data_path.rsplit('/', 1)[1].rsplit('.')[0]]
            benchmark_info['Algorithmus'] += [f'{solver_type} Insert']
            t_0 = time.perf_counter()
            cost = get_insert_solution(str(data_path), solver_type=solver_type)
            t_f = time.perf_counter()
            benchmark_info['Lösungszeit'] += [t_f - t_0]
            benchmark_info['Routenkosten'] += [cost]
            benchmark_info['Relative Differenz zur optimalen Lösung'] += [
                f'{100 * (cost - optimal_cost) / optimal_cost:.2f}%']

        for solver, solver_name in zip([get_sweep_solution, get_savings_solution, get_nearest_neighbour_solution],
                                       ['Sweep', 'Savings', 'Nearest Neighbour']):
            benchmark_info['Benchmark'] += [data_path.rsplit('/', 1)[1].rsplit('.')[0]]
            benchmark_info['Algorithmus'] += [solver_name]
            t_0 = time.perf_counter()
            cost = solver(str(data_path))
            t_f = time.perf_counter()
            benchmark_info['Lösungszeit'] += [t_f - t_0]
            benchmark_info['Routenkosten'] += [cost]
            benchmark_info['Relative Differenz zur optimalen Lösung'] += [
                f'{100 * (cost - optimal_cost) / optimal_cost:.2f}%']

    df = pd.DataFrame(benchmark_info)
    df.to_csv(f'Benchmark_Results/results_benchmark_{benchmark_folder_path.rsplit("/",1)[1]}.csv')


if __name__ == '__main__':
    benchmark_opening_heuristics()
