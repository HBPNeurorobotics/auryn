import pandas as pd
import sys
import numpy as np
import csv


def ras_to_df(filepath, start=0., end=sys.maxint, startseek=0):
    if isinstance(filepath, str):
        if filepath.find('*') > 0:
            import glob
            filenames = glob.glob(filepath)
        else:
            filenames = [filepath]
    else:
        assert hasattr(filepath, '__len__')
        filenames = filepath
    df = pd.DataFrame({'ts': [], 'n_id': []})
    seek = 0
    for filename in filenames:
        with open(filename, 'r') as f:
            f.seek(startseek)
            ret = []
            read = True
            while read:
                line = f.readline()
                sol = np.array(line.split(), dtype=float)
                if sol.size > 0 and sol[0] < end:
                    if start <= sol[0]:
                        ret.append(sol)
                else:
                    read = False
            reta = np.array(ret)
            seek = f.tell()
        if reta.size != 0:
            df = df.append(pd.DataFrame({'ts': reta[:, 0], 'n_id': reta[:, 1]}))
    return df, seek


def mtx_file_to_matrix(filepath):
    with open(filepath, 'r') as f:
        filelines = f.read().splitlines()
        dims = map(int, filelines[2].split())
        weight_matrix = np.zeros((dims[0], dims[1]), dtype=float)
        for line in filelines[3:]:
            words = line.split()
            weight_matrix[int(words[0]) - 1][int(words[1]) - 1] = float(words[2])
    return weight_matrix


def get_weight_stats(path, context, connections=['vh', 'hh', 'ho']):
    reccurent = context['recurrent']
    nh1 = context['nh1']
    stats = {}
    for connection in connections:
        weight_matrix = mtx_file_to_matrix(path.format(connection))
        if reccurent and connection == 'hh':
            stats['h1h2'] = [(np.mean(weight_matrix[:nh1]), np.std(weight_matrix[:nh1]))]
            stats['h2h1'] = [(np.mean(weight_matrix[nh1:]), np.std(weight_matrix[nh1:]))]
        else:
            stats[connection] = [(np.mean(weight_matrix), np.std(weight_matrix))]
    return stats


def read_label_csv(path):
    with open(path, 'rb') as csvfile:
        dict_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
        labels = []
        start_times = []
        end_times = []
        for line in dict_reader:
            labels.append(int(line['class']))
            start_times.append(int(line['startTime_usec']))
            end_times.append(int(line['endTime_usec']))
        assert start_times[0] < start_times[-1]
        assert end_times[0] < end_times[-1]
        return zip(labels, start_times, end_times)
