import os

import pandas as pd
from typing import *


def generate_aggregated_table(aggregated_table_save_path: Union[str, os.PathLike] = 'Aggregierte_Übersicht.csv',
                              benchmark_result_path: Union[
                                  str, os.PathLike] = 'Benchmark_Results/results_benchmark_X.csv',
                              additional_benchmark_information_path: Union[
                                  str, os.PathLike] = 'additional_benchmark_information.csv'):
    """Uses the specified data to generate an aggregated result table for a given benchmark,
    the name of the benchmark has to be present in the column 'Benchmark',
    the depot position type column has to be named 'Depotposition',
     the customer position type column has to be named 'Kundenposition'
     in order for the function to work

    :param aggregated_table_save_path: Path to save the new aggregate table at
    :param benchmark_result_path: Path to the benchmark result csv file
    :param additional_benchmark_information_path: Path to the additional information csv file
    :return:
    """
    aggregated_data = {'Algorithmus': [], 'Depotstandort': [], 'Kundenverteilung': [],
                       'Mittelwert prozentuale Abweichung von optimaler Lösung': [], 'minimale Lösungszeit': []}
    df = pd.read_csv(benchmark_result_path, index_col='Unnamed: 0')
    df_additional = pd.read_csv(additional_benchmark_information_path, index_col='Unnamed: 0')
    df = df.merge(df_additional, on=['Benchmark'])
    for groups, data in df.groupby(['Depotposition', 'Kundenposition']):
        for algo, algo_data in data.groupby('Algorithmus'):
            algo_data['Relative Differenz zur optimalen Lösung'] = algo_data[
                                                                       'Relative Differenz zur optimalen Lösung'].str[
                                                                   :-1].astype(float)
            aggregated_data['Algorithmus'] += [algo]
            aggregated_data['Depotstandort'] += [groups[0]]
            aggregated_data['Kundenverteilung'] += [groups[1]]
            aggregated_data['Mittelwert prozentuale Abweichung von optimaler Lösung'] += [
                algo_data['Relative Differenz zur optimalen Lösung'].mean()]
    df_means = pd.DataFrame(aggregated_data)
    df_means.to_csv(aggregated_table_save_path)
