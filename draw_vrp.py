import os

import numpy as np
import vrplib
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from typing import *
from ast import literal_eval


def get_random_color() -> tuple[float, float, float]:
    return (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))


def get_lines_and_colors(sol: dict[str, Any], ax: plt.Axes) -> Tuple[List[Any], List[Any]]:
    lines = []
    colors = []

    # Generate route lines and assign a random color to each route
    for route in sol['routes']:
        color = get_random_color()
        for _ in route:
            line, = ax.plot([], [], color=color)
            lines.append(line)
            colors.append(color)
        line, = ax.plot([], [], color=color)
        lines.append(line)
        colors.append(color)
    return lines, colors


def get_nested_route_lines(sol: List[List[List[int]]], ax: plt.Axes) -> List[Any]:
    lines = []

    # Generate route lines and assign a random color to each route
    for route in sol:
        for sub_route in route:
            for _ in sub_route:
                line, = ax.plot([], [])
                lines.append(line)
    return lines


def get_route_lines(routes: List[List[Any]], point_mapping: dict[int, Any]) -> np.ndarray:
    route_line_coordinates = []
    for route in routes:
        last_point = 0
        for current_point in route:
            route_line_coordinates.append(([point_mapping[last_point][0], point_mapping[current_point][0]],
                                           [point_mapping[last_point][1], point_mapping[current_point][1]]))
            last_point = current_point
        route_line_coordinates.append(([point_mapping[current_point][0], point_mapping[0][0]],
                                       [point_mapping[current_point][1], point_mapping[0][1]]))
    return np.array(route_line_coordinates)


def get_routes(routes: List[List[Any]], point_mapping: dict[int, Any]) -> List[List[np.ndarray]]:
    all_routes = []
    for route in routes:
        sub_route_coordinates = []
        for sub_route in route:
            route_line_coordinates = []
            last_point = 0
            for current_point in sub_route:
                route_line_coordinates.append(([point_mapping[last_point][0], point_mapping[current_point][0]],
                                               [point_mapping[last_point][1], point_mapping[current_point][1]]))
                last_point = current_point
            route_line_coordinates.append(([point_mapping[current_point][0], point_mapping[0][0]],
                                           [point_mapping[current_point][1], point_mapping[0][1]]))
            sub_route_coordinates.append(np.array(route_line_coordinates))

        all_routes.append(sub_route_coordinates)
    return all_routes


def plot_sequential_solution_from_file(instance_path: Union[str, os.PathLike], solution_path):
    fig, ax = plt.subplots()
    instance = vrplib.read_instance(instance_path)
    solution = vrplib.read_solution(solution_path)

    plt.plot(instance['node_coord'][0][0], instance['node_coord'][0][1], 'o', markersize=5, c='r')
    point_map = {num: list(edge) for num, edge in enumerate(instance['node_coord'])}
    for edge, demand in zip(instance['node_coord'][1:], instance['demand'][1:]):
        ax.plot(edge[0], edge[1], 'o', markersize=1, c='g')

    lines, colors = get_lines_and_colors(solution, ax)
    route_lines = get_route_lines(solution['routes'], point_map)

    def update(num):
        if num < len(route_lines):
            lines[num].set_data(route_lines[num][0], route_lines[num][1])
        return lines

    anim = animation.FuncAnimation(fig, update, frames=len(route_lines), interval=500, blit=True)

    plt.show()


def plot_insertion_solution(instance_path: Union[str, os.PathLike], route_history_path):
    fig, ax = plt.subplots()
    instance = vrplib.read_instance(instance_path)
    with open(route_history_path, 'r', encoding='utf-8') as f:
        route_history = literal_eval(f.read())

    plt.plot(instance['node_coord'][0][0], instance['node_coord'][0][1], 'o', markersize=5, c='r')
    point_map = {num: list(edge) for num, edge in enumerate(instance['node_coord'])}
    for edge, demand in zip(instance['node_coord'][1:], instance['demand'][1:]):
        ax.plot(edge[0], edge[1], 'o', markersize=1, c='g')
    lines = get_nested_route_lines(route_history, ax)
    routes = get_routes(route_history, point_map)
    running_cnt = 0
    cnt_route_mapping = {}
    for num, route in enumerate(routes):
        col = get_random_color()
        for sub_route in route:
            cnt_route_mapping[running_cnt] = (np.concatenate(sub_route, axis=1), col)
            running_cnt += 1

    def update(num):
        try:
            if cnt_route_mapping[num][1] == cnt_route_mapping[num - 1][1]:
                lines[num - 1].set_data([], [])
        except KeyError:
            pass
        lines[num].set_data(cnt_route_mapping[num][0])
        lines[num].set_color(cnt_route_mapping[num][1])
        return lines

    anim = animation.FuncAnimation(fig, update, frames=len(cnt_route_mapping), interval=500, blit=True, repeat=False)
    anim.save(f'animations/{route_history_path.rsplit("/",1)[1].rsplit(".",1)[0]}.gif', writer='pillow')

    plt.show()


if __name__ == '__main__':
    base_path = 'data/E/E-n51-k5'
    instance_path = f"{base_path}.vrp"
    solution_path = f"solutions/Insertion/route_history.txt"
    nn_solution_path = 'solutions/Insertion/test_cheapeast_insertion.sol'
    # plot_sequential_solution_from_file(instance_path, solution_path)
    plot_insertion_solution(instance_path, solution_path)
