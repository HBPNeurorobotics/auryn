import erbp_plotter as plotter
import jaer_data_handler as jhandler
import numpy as np
import os
import glob
import shutil


def generate_video_from_aedat(paths, aedat_version):
    for path in paths:
        os.mkdir('plots/video/pics')
        if aedat_version == 'aedat3':
            ts, x, y, p = jhandler.load_aedat31(path)
        else:
            ts, x, y, p = jhandler.load_jaer(path, version=aedat_version, debug=1)
        ts = np.array(ts)
        if ts[0] > ts[-1]:
            print('HAD TO RESTORE TS ORDER')
            ts = restore_ts_order(ts)
        ts -= min(ts)
        max_ts_in_s = max(ts) * 1e-6
        print(max_ts_in_s)
        framerate = 1. / 60
        file_name = path.split('/')[-1].split('.')[0]
        for i, start in enumerate(np.arange(0, max_ts_in_s, framerate)):
            # plotter.plot_2d_hist_from_aedat31(path, start=start, end=start + framerate,
            #                                  image_title='plots/video/hist{:05d}'.format(i))
            plotter.plot_2d_events_from_aedat(path, start=start, end=start + framerate,
                                              image_title='plots/video/pics/events{:05d}'.format(i), version=aedat_version)
        os.system('ffmpeg -r 60 -f image2 -s 1920x1080 -i plots/video/pics/events%05d.png -vcodec libx264 -crf 15  -pix_fmt yuv420p plots/video/{file_name}_events.mp4'.format(
                file_name=file_name))
        shutil.rmtree('plots/video/pics')


def restore_ts_order(timestamps):
    for i in range(len(timestamps) - 1):
        if timestamps[i] > timestamps[i + 1]:
            timestamps[:i + 1] -= (2 ** 32 * 1e-6)
            return timestamps


generate_video_from_aedat(glob.glob('data/dvs_gesture_split/train/user01_fluorescent4__5.aedat'), 'aedat3')
