import os
import random
import numpy as np
import pandas as pd
import vrplib
from typing import *
from pathlib import Path
import time
from tqdm import tqdm

from typing import List, Any, Dict, Union
import numpy as np


def move_element(base_list: List[Any], obj_idx: int, new_position: int) -> List[Any]:
    """
    Moves an element within a list to a new position.

    :param base_list: The list containing the element to move
    :param obj_idx: The index of the element to move
    :param new_position: The new position for the element
    :return: A new list with the element moved to the new position
    """
    new_list = base_list.copy()
    new_list.insert(new_position, new_list.pop(obj_idx))
    return new_list


def swap_positions(base_list: List[Any], position1: int, position2: int) -> List[Any]:
    """
    Swaps the position of 2 list elements.

    :param base_list: List whose elements will be swapped
    :param position1: First position to swap
    :param position2: Second position to swap
    :return: List with swapped elements
    """
    base_list[position1], base_list[position2] = base_list[position2], base_list[position1]
    return base_list


def calculate_route_length(route: List[int]) -> float:
    """
    Calculates the total length of a route.

    :param route: List of points representing the route
    :return: Total length of the route
    """
    return sum([np.linalg.norm(point - route[num]) for num, point in enumerate(route[1:])])


def get_nearest_neighbour_solution(instance: Dict[str, Any], distance_matrix: np.ndarray) -> List[List[int]]:
    """
    Generates a nearest neighbour solution for the given instance.

    :param instance: Dictionary containing the problem instance data
    :param distance_matrix: Matrix containing the distances between points
    :return: List of routes generated by the nearest neighbour algorithm
    """
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
    return routes


def calculate_cost(solution: Dict[str, Any], distance_matrix: np.ndarray) -> Union[int, float]:
    """
    Calculates the cost of a given solution.

    :param solution: Dictionary containing the solution routes  {'routes':[route for route in solution_routes]}
    :param distance_matrix: Matrix containing the distances between points
    :return: Total cost of the solution
    """
    cost = 0
    distance_matrix = np.round(distance_matrix)
    for route in solution['routes']:
        last = 0
        for edge in route:
            cost += distance_matrix[last, edge]
            last = edge
        cost += distance_matrix[last, 0]
    return cost


def run_cheapest_insertion_loop(points: List[int], capacity_limit: int, demands: List[int],
                                distance_matrix: np.ndarray) -> List[List[int]]:
    """
    Runs the cheapest insertion loop algorithm to generate routes.

    :param points: List of points to be routed
    :param capacity_limit: Capacity limit for each route
    :param demands: List of demands for each point
    :param distance_matrix: Matrix containing the distances between points
    :return: List of routes generated by the cheapest insertion loop algorithm
    """
    routes = []
    while points:
        current_route = [0]
        start = np.argmin(distance_matrix[0, points])
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route += [current_point, 0]
        distances = [distance_matrix[0, current_point], distance_matrix[0, current_point]]

        while capacity < capacity_limit and points:
            insertion_prices = [
                distance_matrix[current_route[num], edge] + distance_matrix[current_route[num + 1], edge] - old_dist
                for edge in points for num, old_dist in enumerate(distances)]
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
        routes += [current_route[1:-1]]
    return routes


def get_cheapest_insert_index(current_edge_numbers: List[int], new_edge: int, distance_matrix: np.ndarray) -> int:
    """
    Finds the cheapest insertion index for a new edge in the current route.

    :param current_edge_numbers: List of current edges in the route
    :param new_edge: The new edge to be inserted
    :param distance_matrix: Matrix containing the distances between points
    :return: The index at which to insert the new edge for the least cost
    """
    return np.argmin(
        np.array([distance_matrix[current_edge, new_edge] + distance_matrix[current_edge_numbers[num], new_edge]
                  - distance_matrix[current_edge_numbers[num], current_edge] for num, current_edge in
                  enumerate(current_edge_numbers[1:])])) + 1


def run_nearest_insertion_loop(points: List[int], capacity_limit: int, demands: List[int],
                               distance_matrix: np.ndarray) -> List[List[int]]:
    """
    Runs the nearest insertion loop algorithm to generate routes.

    :param points: List of points to be routed
    :param capacity_limit: Capacity limit for each route
    :param demands: List of demands for each point
    :param distance_matrix: Matrix containing the distances between points
    :return: List of routes generated by the nearest insertion loop algorithm
    """
    routes = []
    while points:
        current_route = [0]
        start = np.argmin(distance_matrix[0, points])
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route += [current_point, 0]
        while capacity < capacity_limit and points:
            next_edge_index = np.argmin(np.array(
                [np.min([distance_matrix[current_route_index, point] for current_route_index in current_route])
                 for point in points]))
            if capacity + demands[next_edge_index] <= capacity_limit:
                next_edge = points.pop(next_edge_index)
                insertion_index = get_cheapest_insert_index(current_route, next_edge, distance_matrix)
                current_route.insert(insertion_index, next_edge)
                capacity += demands.pop(next_edge_index)
            else:
                break
        routes += [current_route[1:-1]]
    return routes


def run_farthest_insertion_loop(points: List[int], capacity_limit: int, demands: List[int],
                                distance_matrix: np.ndarray) -> List[List[int]]:
    """
    Runs the farthest insertion loop algorithm to generate routes.

    :param points: List of points to be routed
    :param capacity_limit: Capacity limit for each route
    :param demands: List of demands for each point
    :param distance_matrix: Matrix containing the distances between points
    :return: List of routes generated by the farthest insertion loop algorithm
    """
    routes = []
    while points:
        current_route = [0]
        start = np.argmax(distance_matrix[0, points])
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route += [current_point, 0]
        while capacity < capacity_limit and points:
            next_edge_index = np.argmax(np.array(
                [np.max([distance_matrix[current_point_index, point] for current_point_index in current_route])
                 for point in points]))
            if capacity + demands[next_edge_index] <= capacity_limit:
                next_edge = points.pop(next_edge_index)
                insertion_index = get_cheapest_insert_index(current_route, next_edge, distance_matrix)
                current_route.insert(insertion_index, next_edge)
                capacity += demands.pop(next_edge_index)
            else:
                break
        routes += [current_route[1:-1]]
    return routes


def run_random_insertion_loop(points: List[int], capacity_limit: int, demands: List[int],
                              distance_matrix: np.ndarray) -> List[List[int]]:
    """
    Runs the random insertion loop algorithm to generate routes.

    :param points: List of points to be routed
    :param capacity_limit: Capacity limit for each route
    :param demands: List of demands for each point
    :param distance_matrix: Matrix containing the distances between points
    :return: List of routes generated by the random insertion loop algorithm
    """
    routes = []
    while points:
        current_route = [0]
        start = random.randrange(0, len(points))
        capacity = demands.pop(start)
        current_point = points.pop(start)
        current_route += [current_point, 0]
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
        routes += [current_route]
    return routes


def get_insert_solution(instance: Dict[str, Any], distance_matrix: np.ndarray,
                        solver_type: str = 'cheapest'):
    """
    Generates routes using a specified insertion heuristic.

    :param instance: Dictionary containing the VRP instance data
    :param distance_matrix: Matrix containing the distances between points
    :param solver_type: Type of insertion heuristic to use ('cheapest', 'nearest', 'farthest', 'random')
    :return: List of routes generated by the specified insertion heuristic
    """
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
    return routes


def get_savings_solution(instance: Dict[str, Any], distance_matrix: np.ndarray):
    """
    Generates routes using the savings algorithm.

    :param instance: Dictionary containing the VRP instance data
    :param distance_matrix: Matrix containing the distances between points
    :return: List of routes generated by the savings algorithm
    """
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
        _i, _j = saving[1]
        try:
            first_route = list(filter(lambda x: x[0] == _i or x[-1] == _i, routes))[0]
            second_route = list(filter(lambda x: x[0] == _j or x[-1] == _j, routes))[0]
        except IndexError:
            continue
        if first_route == second_route or first_route == second_route[::-1]: continue
        if np.sum([demands[num] for num in first_route] + [demands[num] for num in second_route]) <= capacity_limit:
            if _i == first_route[0]:
                first_route = first_route[::-1]
            if _j == second_route[-1]:
                second_route = second_route[::-1]

            routes = list(
                map(lambda x: first_route + second_route if x == first_route or x == first_route[::-1] else x, routes))
            try:
                routes.remove(second_route)
            except ValueError:
                routes.remove(second_route[::-1])
    return routes


def sort_by_polar_angle(points):
    """
    Sorts points by their polar angle relative to the origin.

    :param points: List of points to be sorted
    :return: List of points sorted by their polar angle
    """

    def key(x):
        x = x[0]
        atan = np.arctan2(x[1], x[0])
        return atan if atan >= 0 else 2 * np.pi + atan

    return sorted(points, key=key)


def get_sweep_solution(instance: Dict[str, Any], distance_matrix: np.ndarray,
                       solver_type: str = 'cheapest'):
    """
    Generates routes using the sweep algorithm combined with an insertion heuristic.

    :param instance: Dictionary containing the VRP instance data
    :param distance_matrix: Matrix containing the distances between points
    :param solver_type: Type of insertion heuristic to use ('cheapest', 'nearest', 'farthest', 'random')
    :return: List of routes generated by the sweep algorithm combined with the specified insertion heuristic
    """
    depot_location = instance['node_coord'][0]
    points = list(instance['node_coord'])[1:]
    capacity_limit = instance['capacity']
    demands = list(instance['demand'])
    angles = sort_by_polar_angle([(point - depot_location, num + 1) for num, point in enumerate(points)])

    insertion_loop_mapping = {'cheapest': run_cheapest_insertion_loop, 'nearest': run_nearest_insertion_loop,
                              'farthest': run_farthest_insertion_loop, 'random': run_random_insertion_loop}
    assert insertion_loop_mapping.get(solver_type), AssertionError(f'Solver does not exist, you gave {solver_type},'
                                                                   f' but only {list(insertion_loop_mapping.keys())} '
                                                                   f'are implemented solvers')
    results = {}
    for start_point, num_route in angles:
        num_route -= 1
        copied_angles = angles.copy()
        copied_angles = copied_angles[num_route:] + copied_angles[:num_route]
        routes = []
        copied_demands = demands.copy()

        while copied_angles:

            route = []
            while copied_angles and sum([copied_demands[num] for num in route]) + copied_demands[
                copied_angles[0][1]] <= capacity_limit:
                route += [copied_angles.pop(0)[1]]
            routes += [route]

        for num, route in enumerate(routes):
            route = insertion_loop_mapping[solver_type](points=route,
                                                        capacity_limit=capacity_limit, demands=copied_demands,
                                                        distance_matrix=distance_matrix)
            routes[num] = route[0]
        results[calculate_cost({'routes': routes}, distance_matrix)] = routes

    return results[min(results)]


def get_relocated_route(route: List[np.ndarray]) -> List[np.ndarray]:
    """
    Relocates elements in the route to find the shortest possible route.

    :param route: List of nodes in the route
    :return: Modified route with relocated elements for optimal distance
    """
    if len(route) <= 1: return route
    possible_routes = [[calculate_route_length(move_element(route, i, j)), (i, j)] for i, _ in enumerate(route) for
                       j, _ in enumerate(route) if i < j]
    best_route = min(possible_routes, key=lambda x: x[0])
    return move_element(route, best_route[1][0], best_route[1][1])


def get_exchanged_route(route: List[np.ndarray]) -> List[np.ndarray]:
    """
    Swaps positions of elements in the route to find the shortest possible route.

    :param route: List of nodes in the route
    :return: Modified route with swapped elements for optimal distance
    """
    if len(route) <= 1: return route
    possible_routes = [[calculate_route_length(swap_positions(route, i, j)), (i, j)] for i, _ in enumerate(route) for
                       j, _ in enumerate(route) if i < j]
    best_route = min(possible_routes, key=lambda x: x[0])
    return swap_positions(route, best_route[1][0], best_route[1][1])


def finalize_results(routes: List[List[int]], distance_matrix: np.ndarray, instance_name: str):
    """
    Calculates the cost of routes and writes the solution to a file.

    :param routes: List of routes
    :param distance_matrix: Matrix containing the distances between points
    :param instance_name: Name of the VRP instance
    :return: Total cost of the routes
    """
    cost = calculate_cost({'routes': routes}, distance_matrix)
    vrplib.write_solution(f'solutions/NearestNeighbour/{instance_name}.sol', routes,
                          {'cost': cost})
    return cost


def benchmark_opening_heuristics(benchmark_folder_path: Union[os.PathLike, str] = r'data/Seminar_Arbeit_Subset'):
    """
    Benchmarks various opening heuristics on a set of VRP instances.

    :param benchmark_folder_path: Path to the folder containing VRP benchmark instances
    """
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
            routes = get_insert_solution(instance=benchmark, solver_type=solver_type,
                                         distance_matrix=dist_mat)
            t_f = time.perf_counter()
            cost = finalize_results(routes, instance_name=benchmark_name, distance_matrix=dist_mat)
            benchmark_info['Lösungszeit'] += [t_f - t_0]
            benchmark_info['Routenkosten'] += [cost]
            benchmark_info['Relative Differenz zur optimalen Lösung'] += [
                f'{100 * (cost - optimal_cost) / optimal_cost:.2f}%']

        for solver_type in ['cheapest', 'nearest', 'farthest', 'random']:
            benchmark_info['Benchmark'] += [benchmark_name]
            benchmark_info['Algorithmus'] += [f'Sweep + {solver_type} Insert']
            t_0 = time.perf_counter()
            routes = get_sweep_solution(instance=benchmark, solver_type=solver_type,
                                        distance_matrix=dist_mat)
            t_f = time.perf_counter()
            cost = finalize_results(routes, instance_name=benchmark_name, distance_matrix=dist_mat)

            benchmark_info['Lösungszeit'] += [t_f - t_0]
            benchmark_info['Routenkosten'] += [cost]
            benchmark_info['Relative Differenz zur optimalen Lösung'] += [
                f'{100 * (cost - optimal_cost) / optimal_cost:.2f}%']

        for solver, solver_name in zip([get_savings_solution, get_nearest_neighbour_solution],
                                       ['Savings', 'Nearest Neighbour']):
            benchmark_info['Benchmark'] += [benchmark_name]
            benchmark_info['Algorithmus'] += [solver_name]
            t_0 = time.perf_counter()
            routes = solver(instance=benchmark, distance_matrix=dist_mat)
            t_f = time.perf_counter()
            cost = finalize_results(routes, instance_name=benchmark_name, distance_matrix=dist_mat)
            benchmark_info['Lösungszeit'] += [t_f - t_0]
            benchmark_info['Routenkosten'] += [cost]
            benchmark_info['Relative Differenz zur optimalen Lösung'] += [
                f'{100 * (cost - optimal_cost) / optimal_cost:.2f}%']

    df = pd.DataFrame(benchmark_info)
    df.to_csv(f'Benchmark_Results/results_benchmark_{benchmark_folder_path.rsplit("/", 1)[1]}.csv')


def get_distance_matrix(points: List[np.ndarray]) -> np.ndarray:
    """
    Calculates the distance matrix for a list of points.

    :param points: List of points (coordinates)
    :return: Matrix containing the distances between points
    """
    dist_mat = np.zeros((len(points), len(points)))
    for i in range(dist_mat.shape[0]):
        for j in range(dist_mat.shape[0]):
            dist_mat[i, j] = np.linalg.norm(points[i] - points[j])
    return dist_mat
