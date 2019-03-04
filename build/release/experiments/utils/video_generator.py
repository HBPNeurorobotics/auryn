from erbp_plotter import Plotter
import pandas as pd
import datetime
import jaer_data_handler as jhandler
import numpy as np
import os
import glob
import shutil
import errno
import argparse

plotter = Plotter('/tmp')

def parse_args():
    parser = argparse.ArgumentParser(description='AER video generator')
    parser.add_argument('--input', type=str, default='data/dvs_gesture_split/train/user01_fluorescent4__5.aedat',
                        help='path to input aedat')
    parser.add_argument('--output', type=str, default='',
                        help='path to output'),
    parser.add_argument('--remove_pics', type=bool, default=True,
                        help='remove pics from tmp folder')


    return parser.parse_args()

args = parse_args()

def generate_video_from_aedat(input_path, output_path, aedat_version='aedat3', remove_tmp_pics=True, event_amount=1000, attention_window=True):
    file_name = input_path.split('/')[-1].split('.')[0]
    tmp_folder_pics = os.path.join('/tmp/scripts/plots/{}'.format(file_name))

    if output_path == "":
        output_path = file_name + '.mp4'

    output_folder = os.path.dirname(output_path)

    try:
        os.mkdir(tmp_folder_pics)
        if output_folder is not "":
            os.mkdir(output_folder)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise

    if aedat_version == 'aedat3':
        timestamps, xaddr, yaddr, pol = jhandler.load_aedat31(input_path, debug=0)
    else:
        timestamps, xaddr, yaddr, pol = jhandler.load_jaer(input_path, version='aedat', debug=0)
        timestamps = np.array(timestamps)
        if timestamps[0] > timestamps[-1]:
            print('HAD TO RESTORE TS ORDER')
            timestamps = self.restore_ts_order(timestamps)
        timestamps -= min(timestamps)
    df = pd.DataFrame({'ts': timestamps, 'x': xaddr, 'y': yaddr, 'p': pol})
    df.ts = df.ts * 1e-6

    framerate = 30
    dt = 1. / framerate
    max_ts_in_s = df.ts.max()
    # compute centroids and add them to the event df
    centroids = df.loc[:, ['x', 'y']].rolling(window=event_amount, min_periods=1).median().astype(int)
    centroids = centroids.rename(columns={"x": "centroid_x", "y": "centroid_y"})

    df = pd.merge(df, centroids, how='outer', left_index=True, right_index=True)

    for i, start in enumerate(np.arange(0, max_ts_in_s, dt)):
        end=start + dt
        if end > max(df.ts):
            end = max(df.ts)
        print(start, end)
        current_df = df[(df.ts >= start) & (df.ts <= end)]
        current_centroid = current_df[['centroid_x', 'centroid_y']].mean()
        plotter.plot_2d_events_from_df(current_df, centroid=current_centroid,
                                       plot_title='Events from {:0.2f}s to {:.2f}s'.format(start, end),
                                       image_title='{}/events{:05d}'.format(file_name, i))

    # animate the images with ffmpeg
    os.system('ffmpeg -y -r {framerate} -f image2 -s 1280x720 -i {tmp_folder}/events%05d.png -vcodec libx264 -crf 15 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p {output}'.format(
        framerate=framerate,
        output=output_path,
        tmp_folder=tmp_folder_pics))
    if remove_tmp_pics:
        shutil.rmtree(tmp_folder_pics)


def restore_ts_order(timestamps):
    for i in range(len(timestamps) - 1):
        if timestamps[i] > timestamps[i + 1]:
            timestamps[:i + 1] -= (2 ** 32 * 1e-6)
            return timestamps


generate_video_from_aedat(args.input,
                          args.output,
                          remove_tmp_pics=args.remove_pics)
