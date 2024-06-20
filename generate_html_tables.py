import os
from typing import Union

import pandas as pd


def generate_html_tables(
        result_path: Union[str, os.PathLike] = 'Benchmark_Results/results_benchmark_Seminar_Arbeit_Subset.csv'):
    """Uses the benchmark results to write a html table for each separate benchmark instance.
    Saves tables inside of Benchmark_Results/tables

    :param result_path: Path the results are located at
    :return: None
    """
    df = pd.read_csv(result_path, index_col='Unnamed: 0')
    for algo in df.groupby('Benchmark'):
        algo[1].drop(columns='Benchmark').to_html(f'Benchmark_results/tables/{algo[0]}.html', encoding='utf-8',
                                                  index=False)
        with open(f'Benchmark_results/tables/{algo[0]}.html', 'r') as f:
            code = "<meta charset='utf-8' >\n" + f.read()
        with open(f'Benchmark_results/tables/{algo[0]}.html', 'w') as f:
            f.write(code)
