import os

import pandas as pd
import requests
from pathlib import Path
from typing import *


def scrape_additional_benchmark_information(benchmark_folder_path: Union[str, os.PathLike] = 'data/X',
                                             additional_information_save_path: Union[
                                                 str, os.PathLike] = 'additional_benchmark_information.csv'):
    base_url = 'http://vrp.galgos.inf.puc-rio.br/index.php/en/plotted-instances?data='
    data = {'Benchmark': [], 'Maximale Weglänge': [], 'Depotposition': [], 'Kundenposition': []}
    separator = "</strong>\r\n\t"
    for file in Path(benchmark_folder_path).glob('**/*.vrp'):
        benchmark_name = file.name[:-4]
        resp = requests.get(base_url + benchmark_name)
        text = resp.text
        data['Benchmark'] += [benchmark_name]
        words = [r'Upper Bound (UB):', 'Root Positioning:', 'Customer Positioning:']
        for word, fieldname in zip(words, list(data.keys())[1:]):
            split_word = word + separator
            idx = text.find(split_word)
            word = text[idx + len(split_word):]
            idx2 = word.find('</li>')
            data[fieldname] += [word[:idx2].strip()]
    df = pd.DataFrame(data)
    df.to_csv(additional_information_save_path)
