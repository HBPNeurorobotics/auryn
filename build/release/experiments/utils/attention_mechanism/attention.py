import numpy as np

import pandas as pd
from IPython import embed
from tqdm import tqdm

from online_median_heap import OnlineMedianFinder


def get_attention_df_offline(df, attention_window_time, attention_window_size, attention_mechanism):
    dfs = []
    half_att_win = int(attention_window_size / 2)

    for i, time_window in enumerate(np.arange(0.0, max(df.ts), attention_window_time)):
        event_slice = df.loc[(df.ts >= time_window) & (df.ts < time_window + attention_window_time)]
        centroid_x, centroid_y = get_centroid_of_df(event_slice, half_att_win, False, attention_mechanism)
        event_slice = get_events_in_window(centroid_x, centroid_y, attention_window_size, event_slice)
        event_slice = shift_for_attention(event_slice, centroid_x, centroid_y)
        calc_n_id(attention_window_size, event_slice)
        if event_slice.size > 0:
            min_n_id = event_slice['n_id'].min()
            assert min_n_id > 0, 'n_id smaller than zero'
            max_n_id = event_slice['n_id'].max()
            assert max_n_id <= attention_window_size * attention_window_size, 'n_id greater than inputs size'
        dfs.append(event_slice)
    return pd.concat(dfs)


def get_attention_df_rolling(df, event_amount, attention_window_size):
    centroids = df.loc[:, ['x', 'y']].rolling(window=event_amount, min_periods=1).median().astype(int)
    df = take_window_events(attention_window_size, centroids, df)
    return df


def take_window_events(attention_window_size, centroids, df):
    df.loc[:, ['x', 'y']] -= centroids - int(attention_window_size / 2)

    df = df.loc[(df.x >= 0) & (df.x < attention_window_size) & (df.y >= 0) & (df.y < attention_window_size)].copy()
    calc_n_id(attention_window_size, df)
    return df


# deprecated
def get_attention_df_online(df, event_amount):
    embed()
    omf_x = OnlineMedianFinder()
    omf_y = OnlineMedianFinder()
    x = []
    y = []
    for i, event in enumerate(tqdm(df.itertuples())):
        omf_x.add_element(event.x)
        omf_y.add_element(event.y)
        x.append(event.x - omf_x.currentMedian)
        y.append(event.y - omf_y.currentMedian)
        # print(len(omf_y.smallElements)+len(omf_y.bigElements))
        if i > event_amount:
            try:
                omf_x.remove_element(df.iloc[i - event_amount].x)
                omf_y.remove_element(df.iloc[i - event_amount].y)
            except ValueError:
                print('x', omf_x.currentMedian, omf_x.smallElements.getheap(), omf_x.bigElements.getheap(),
                      df.iloc[i - event_amount].x)
                print('y', omf_y.currentMedian, omf_y.smallElements.getheap(), omf_y.bigElements.getheap(),
                      df.iloc[i - event_amount].y)
    return pd.DataFrame({'x': x, 'y': y, 'ts': df.ts, 'p': df.p})


def get_centroid_of_df(event_slice, half_att_win, clip, attention_mechanism):
    if clip:
        centroid_x = int(np.clip(attention_mechanism(event_slice.x), half_att_win, 127 - half_att_win) - half_att_win)
        centroid_y = int(np.clip(attention_mechanism(event_slice.y), half_att_win, 127 - half_att_win) - half_att_win)
    else:
        centroid_x = int(attention_mechanism(event_slice.x) - half_att_win)
        centroid_y = int(attention_mechanism(event_slice.y) - half_att_win)
    return centroid_x, centroid_y


def get_events_in_window(x, y, attention_window_size, event_slice):
    return event_slice.loc[
        (event_slice.x >= x) & (event_slice.x < x + attention_window_size) & (event_slice.y >= y) & (
                event_slice.y < y + attention_window_size)]


def shift_for_attention(event_slice, median_x, median_y):
    event_slice.loc[:, 'x'] -= median_x
    event_slice.loc[:, 'y'] -= median_y
    return event_slice


def calc_n_id(window_size, event_slice):
    event_slice.loc[:, 'n_id'] = (event_slice.y * window_size) + event_slice.x + 1
