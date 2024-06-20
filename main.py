from calculate_solution import benchmark_opening_heuristics
from generate_html_tables import generate_html_tables
from typing import Union
import os


def main(data_path: Union[str, os.PathLike]):
    """Runs the benchmark tests for all files inside the folder located at data_path, generates an output csv
    located at Benchmark_Results and a html table for each separate benchmark instance located at
    Benchmark_Results/tables

    :param data_path: Path to the folder containing the benchmark files
    :return: None
    """
    result_path = f'Benchmark_Results/results_benchmark_{data_path.rsplit("/", 1)[1]}.csv'
    benchmark_opening_heuristics(data_path)
    generate_html_tables(result_path)


if __name__ == '__main__':
    main('data/Seminar_Arbeit_Subset')
