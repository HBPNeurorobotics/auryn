import erbp_plotter as plotter
import jaer_data_handler as jhandler
import numpy as np


def generate_video_from_aedat(path):
    ts, x, y, p = jhandler.load_aedat31(path)
    max_ts_in_s = max(ts) * 1e-6
    framerate = 1. / 60
    for i, start in enumerate(np.arange(0, max_ts_in_s, framerate)):
        plotter.plot_2d_from_aedat31(path, start=start, end=start + framerate, image_title='plots/video/image{:05d}'.format(i))


generate_video_from_aedat('data/dvs_gesture_split/train/user01_fluorescent_led11__11.aedat')
