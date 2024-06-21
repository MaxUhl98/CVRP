import os
from typing import Union

import pandas as pd


def fix_table_encoding(path_to_table: Union[str, os.PathLike]):
    with open(path_to_table, 'r') as f:
        code = "<meta charset='utf-8' >\n" + f.read()
    with open(path_to_table, 'w') as f:
        f.write(code)


def generate_benchmark_html_tables(
        result_path: Union[str, os.PathLike] = 'Benchmark_Results/results_benchmark_Seminar_Arbeit_Subset.csv'):
    """Uses the benchmark results to write a html table for each separate benchmark instance.
    Saves tables inside of Benchmark_Results/tables

    :param result_path: Path the results are located at
    :return: None
    """
    df = pd.read_csv(result_path, index_col='Unnamed: 0')
    for algo in df.groupby('Benchmark'):
        save_path = f'Benchmark_results/tables/{algo[0]}.html'
        algo[1].drop(columns='Benchmark').to_html(save_path, encoding='utf-8',
                                                  index=False)
        fix_table_encoding(save_path)


def generate_aggregated_html_tables(
        aggregated_table_path: Union[str, os.PathLike] = 'Benchmark_Results/Aggregierte_Übersicht.csv'):
    df = pd.read_csv(aggregated_table_path, index_col='Unnamed: 0')
    for group_names, group in df.groupby(['Depotstandort', 'Kundenverteilung']):
        save_path = f'Benchmark_Results/aggregated_tables/{"_".join(list(group_names))}.html'
        group.drop(columns=['Depotstandort', 'Kundenverteilung']).to_html(save_path,
                                                                          encoding='utf-8', index=False)
        fix_table_encoding(save_path)
