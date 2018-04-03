import jaer_data_handler as jhandler
import file_io as fio
import os
import glob
import pandas as pd


def convert_gesture_ds(dest_path):
    source_path = 'data/dvs_gesture'
    os.system('mkdir {dest}'.format(dest=dest_path))
    os.system('mkdir {dest}/noise_samples'.format(dest=dest_path))
    for file in ['gesture_mapping.csv', 'trials_to_test.txt', 'trials_to_train.txt']:
        os.system('cp {source}/{file} {dest}/{file}'.format(source=source_path, dest=dest_path, file=file))
    for aedat in glob.glob('{source}/user*_*.aedat'.format(source=source_path)):
        split_aedat(aedat, dest_path)


def split_aedat(aedat, dest_path):
    timestamps, xaddr, yaddr, pol = jhandler.load_aedat31(aedat)
    df = pd.DataFrame({'ts': timestamps, 'x': xaddr, 'y': yaddr, 'p': pol})
    aedat_name = aedat.split('.')[0]
    label_times = fio.read_label_csv('{aedat_name}_labels.csv'.format(aedat_name=aedat_name))
    if df.ts[0] < label_times[0][1]:
        label_times.append((0, df.ts[0], label_times[0][1]))
    for l1, l2 in zip(label_times, label_times[1:]):
        label_times.append((0, l1[2], l2[1]))
    a_n = aedat_name.split('/')[-1]
    for i, l_t in enumerate(label_times):
        df_slice = df[(l_t[1] < df.ts) & (df.ts < l_t[2])]
        df_slice = df_slice.copy()
        df_slice.reset_index(drop=True, inplace=True)

        if l_t[0] == 0:
            new_file_path = '{dest}/noise_samples/{a_n}{i}__{label}.aedat'.format(dest=dest_path, a_n=a_n, i=i,
                                                                    label=l_t[0])
        else:
            new_file_path = '{dest}/{a_n}{i}__{label}.aedat'.format(dest=dest_path, a_n=a_n, i=i,
                                                                    label=l_t[0])
        header = '#!AER-DAT3.1\n#Format: RAW\n#Source 1: DVS128\n#!END-HEADER\n'
        jhandler.write_to_aedat31(new_file_path, header, df_slice)


convert_gesture_ds('data/dvs_gesture_split')
