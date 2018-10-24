import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import file_io as fio
import jaer_data_handler as jhandler
import pandas as pd
from matplotlib.lines import Line2D
import glob


def plot_2d_input_ras(path, dimension, start=0, end=sys.maxint):
    if start > end:
        print('start time has to be smaller than end time.')
        return
    res = dimension * dimension
    stepsize = 3000
    bucket = np.zeros((32, 32), dtype=int)
    labels = np.array([])
    seek = 0
    for i in xrange(end // stepsize + 1):
        local_start = start + i * stepsize
        local_end = (i + 1) * stepsize
        if end < local_end:
            local_end = end
        df, seek = fio.ras_to_df("inputs/{}/input.ras".format(path), local_start, local_end, seek)
        if df.ts.values.size == 0:
            break
        label_df = df.loc[df.n_id >= res]
        labels = np.unique(np.append(labels, label_df.n_id.unique() - res)).astype(int)
        df2 = df.loc[df.n_id < res]
        for _, n_id in df2.n_id.iteritems():
            bucket[int(n_id // dimension)][int(n_id % dimension)] += 1
        del df
        del df2
        del label_df

    # np.savetxt('plots/{path}_ras_input_{start}_to_{end}.txt'.format(path=path, start=start, end=end), bucket)
    plot_heat_map(bucket, 'Spike count from {}s to {}s (labels: {})'.format(start, end, labels))


def plot_2d_from_txt(path, title, start=0, end=9223372036854775807):
    bucket = np.loadtxt('plots/{path}_ras_input_{start}_to_{end}.txt'.format(path=path, start=start, end=end))
    plot_heat_map(bucket, title)


def plot_2d_hist_from_aedat31(pathname, start=0, end=sys.maxint, image_title=''):
    paths = glob.glob(pathname)
    bucket = np.zeros((128, 128), dtype=int)
    for path in paths:
        timestamps, xaddr, yaddr, pol = jhandler.load_aedat31(path, debug=1)
        df = pd.DataFrame({'ts': timestamps, 'x': xaddr, 'y': yaddr, 'p': pol})
        df.ts = df.ts * 1e-6
        if end > max(df.ts):
            end = max(df.ts)
        df = df[(df.ts >= start) & (df.ts <= end)]
        for event in df.itertuples():
            bucket[event.y][event.x] += 1
    plot_heat_map(bucket, 'Spatial event distribution - label 1'.format(start, end), save=True, image_title=image_title,
                  dynamic_v=True)


def plot_2d_events_from_aedat(path, start=0, end=sys.maxint, image_title='', version='aedat3'):
    if version == 'aedat3':
        timestamps, xaddr, yaddr, pol = jhandler.load_aedat31(path, debug=0)
    else:
        timestamps, xaddr, yaddr, pol = jhandler.load_jaer(path, version='aedat', debug=0)
        timestamps = np.array(timestamps)
        if timestamps[0] > timestamps[-1]:
            print('HAD TO RESTORE TS ORDER')
            timestamps = restore_ts_order(timestamps)
        timestamps -= min(timestamps)
    df = pd.DataFrame({'ts': timestamps, 'x': xaddr, 'y': yaddr, 'p': pol})
    df.ts = df.ts * 1e-6
    if end > max(df.ts):
        end = max(df.ts)
    df = df[(df.ts >= start) & (df.ts <= end)]
    bucket = np.zeros((128, 128), dtype=int)
    for event in df.itertuples():
        if event.p == 1:
            bucket[event.y][event.x] += 1
        elif event.p == 0:
            bucket[event.y][event.x] -= 1
    bucket[bucket > 0] = 1
    bucket[bucket < 0] = -1
    plot_heat_map(bucket, 'Events from {:0.2f}s to {:.2f}s'.format(start, end), save=True, image_title=image_title,
                  show_cbar=False, vmin=-1, vmax=1)


def restore_ts_order(timestamps):
    for i in range(len(timestamps) - 1):
        if timestamps[i] > timestamps[i + 1]:
            timestamps[:i + 1] -= (2 ** 32 * 1e-6)
            return timestamps


def plot_heat_map(bucket, plot_title, save=False, image_title='', show_cbar=True, vmin=0, vmax=10, dynamic_v=False):
    plt.clf()
    fig, ax = plt.subplots()
    if dynamic_v:
        cax = ax.imshow(bucket, cmap='viridis', interpolation='nearest')
    else:
        cax = ax.imshow(bucket, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    #ax.set_title(plot_title)
    if show_cbar:
        cbar = fig.colorbar(cax)
    else:
        custom_lines = [Line2D([0], [0], color=plt.cm.viridis(1.), lw=0, marker='s'),
                        Line2D([0], [0], color=plt.cm.viridis(-1.), lw=0, marker='s')]
        plt.legend(custom_lines, ['ON', 'OFF'], handlelength=0.5, borderpad=0.5, framealpha=0.5)
    if save:
        plt.savefig('{}.png'.format(image_title), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')


def plot_weight_matrix(path, connections=['vh', 'hh', 'ho'], save=False):
    num_subplots = len(connections)
    for i, connection in enumerate(connections):
        weight_matrix = fio.mtx_file_to_matrix(path.format(connection))
        plt.subplot(num_subplots, 1, i + 1)
        plt.gca().set_title(connection)
        cax = plt.imshow(weight_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
        cbar = plt.colorbar(cax)
    plt.tight_layout()
    if save:
        plt.savefig('plots/weight_matrix_{}_{}.png'.format(connections, time.time()), dpi=300)
    else:
        plt.show()
    plt.close('all')


def plot_ras_spikes(pathinput, start, end, layers=['vis', 'hid', 'out'], res=sys.maxint, number_of_classes=10,
                    save=False, input_att_window=False, att_win_input_size=128 * 2):
    title = 'Spike times'
    plt.title(title)
    counter = 1
    num_plots = len(layers)
    if 'vis' in layers:
        num_plots += 1
        if input_att_window:
            num_plots += 1

    for layer in layers:
        path = pathinput.format(layer)
        data_df, seek = fio.ras_to_df(path, start, end)
        ax1 = plt.subplot(num_plots, 1, counter)
        ax1.get_yaxis().set_label_coords(-0.1, 0.5)
        if layer == 'vis':
            label_df = data_df.loc[data_df.n_id >= res]
            data_df = data_df.loc[data_df.n_id < res]
            for i in xrange(number_of_classes):
                plt.plot((start, end), (i, i), 'r--', linewidth=0.1, alpha=1)
            plt.plot(label_df.ts.values, label_df.n_id.values - res, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                     markersize=1, alpha=0.1)
            plt.ylabel("Label")
            ax1.xaxis.set_major_locator(plt.NullLocator())
            counter += 1
            x1, x2, y1, y2 = plt.axis()
            plt.axis((start, end, y1, y2))
            if input_att_window:
                data_neuron_size = res - att_win_input_size
                att_pos_df = data_df.loc[data_df.n_id >= data_neuron_size]
                data_df = data_df.loc[data_df.n_id < data_neuron_size]
                ax1 = plt.subplot(num_plots, 1, counter, sharex=ax1)
                ax1.get_yaxis().set_label_coords(-0.1, 0.5)
                plt.plot(att_pos_df.ts.values, att_pos_df.n_id.values, linestyle='None', marker=u',',
                         color=[0, 0, 1, 1])
                plt.ylabel("Attention")
                ax1.xaxis.set_major_locator(plt.NullLocator())
                counter += 1
                x1, x2, y1, y2 = plt.axis()
                plt.axis((start, end, y1, y2))
            ax1 = plt.subplot(num_plots, 1, counter)
            ax1.get_yaxis().set_label_coords(-0.1, 0.5)
            plt.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u',', color=[0, 0, 1, 1])
            plt.ylabel("Input")
            ax1.xaxis.set_major_locator(plt.NullLocator())
        elif layer == 'hid':
            plt.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u',', color=[0, 0, 1, 1])
            plt.plot((start, end), (200, 200), 'r--', linewidth=0.1, alpha=1)
            plt.ylabel("Hidden")
            ax1.xaxis.set_major_locator(plt.NullLocator())
        elif layer == 'out':
            for i in xrange(number_of_classes):
                plt.plot((start, end), (i, i), 'r--', linewidth=0.1, alpha=1)
            plt.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                     markersize=1, alpha=0.2)
            plt.ylabel("Output")
        elif layer == 'err1':
            for i in xrange(number_of_classes):
                plt.plot((start, end), (i, i), 'r--', linewidth=0.1, alpha=1)
            plt.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                     markersize=1, alpha=0.2)
            path = pathinput.format(layer.replace('1', '2'))
            data_df, seek = ras_to_df(path, start, end)
            plt.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u'|', color=[1, 0, 0, 1],
                     markersize=1, alpha=0.2)
            plt.ylabel("Error")
        print(data_df.ts.values.size)
        counter += 1
        x1, x2, y1, y2 = plt.axis()
        plt.axis((start, end, y1, y2))

    plt.xlabel("Time in seconds")
    plt.tight_layout()

    if save:
        plt.savefig('plots/{}_{}_{}_{}.png'.format(start, end, layers, time.time()), dpi=300)
    else:
        plt.show()
    plt.close('all')


def plot_weight_stats(stats, save=False):
    num_subplots = len(stats.keys())
    if num_subplots == 3:
        keys = ['vh', 'hh', 'ho']
    elif num_subplots == 4:
        keys = ['vh', 'h1h2', 'h2h1', 'ho']
    for i, key in enumerate(keys):
        stat = stats[key]
        plt.subplot(num_subplots, 1, i + 1)
        plt.title(key)
        length = len(stat)
        plt.plot((0, length), (0, 0), 'r--', linewidth=0.5)
        plt.errorbar(range(length), [i[0] for i in stat], [i[1] for i in stat], linestyle='None', marker='o')
        plt.ylabel('weight')
    plt.xlabel('epoch')
    plt.tight_layout()
    if save:
        plt.savefig('plots/weight_stats_{}.png'.format(time.time()), dpi=300)
    else:
        plt.show()
    plt.close('all')


def plot_output_weights_over_time(output_weights, save=False):
    plt.clf()
    plt.plot(output_weights, alpha=0.1)
    if save:
        plt.savefig('plots/output_weights_{}.png'.format(time.time()), dpi=300)
    else:
        plt.show()
    plt.close('all')


def plot_accuracy_rate_first(acc_hist, save=False):
    #x = [i[0] for i in acc_hist]
    y_rate = [i[1][0] for i in acc_hist]
    y_first = [i[1][1] for i in acc_hist]
    plt.plot(y_rate, marker='o')
    plt.plot(y_first, marker='o', color='r')
    #plt.title('classification accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy in percent')
    plt.legend(['rate', 'first'])
    if save:
        plt.savefig('plots/acc_hist_{}.png'.format(time.time()), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')


def plot_weight_histogram(path, nh1, connections=['vh', 'hh', 'ho'], save=False):
    num_subplots = len(connections)
    for i, connection in enumerate(connections):
        weight_matrix = fio.mtx_file_to_matrix(path.format(connection))
        plt.subplot(num_subplots, 1, i + 1)
        plt.gca().set_title(connection)
        if connection == 'hh':
            weight_matrix = weight_matrix[:, nh1:]
        elif connection == 'ho':
            weight_matrix = weight_matrix[nh1:]
        weight_matrix = weight_matrix.flatten()
        cax = plt.hist(weight_matrix, 300)
    plt.tight_layout()
    if save:
        plt.savefig('plots/weight_histogram_{}_{}.png'.format(connections, time.time()), dpi=300)
    else:
        plt.show()
    plt.close('all')


def plot_confusion_matrix(df_confusion, save=False):
    plt.imshow(df_confusion, cmap=plt.cm.viridis, vmax=1.0)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))

    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.set_ticks_position('top')
    if save:
        plt.savefig('plots/confusion_matrix_{}.png'.format(time.time()), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')


def plot_weight_convolution(path, nh1, nc, connections=['vh', 'hh', 'ho'], save=False):
    weight_matrices = {}
    for connection in connections:
        weight_matrix = mtx_file_to_matrix("{path}/fwmat_{connection}.mtx".format(path=path, connection=connection))
        if connection == 'vh':
            weight_matrix = weight_matrix[:32 * 32]
        elif connection == 'hh':
            weight_matrix = weight_matrix[:, nh1:]
        elif connection == 'ho':
            weight_matrix = weight_matrix[nh1:]
        weight_matrices[connection] = weight_matrix
    conv_matrix = calc_conv(connections, weight_matrices)  # .reshape(32,32,12)
    # conv_matrix = np.array(map(lambda x: np.argmax(x), conv_matrix)).reshape(32,32)
    for i in range(nc):
        plt.clf()
        conv_label = np.array([item[i] for item in conv_matrix]).reshape(32, 32)

        fig, ax = plt.subplots()
        ax.set_title('Weight convolution for label {}'.format(i))
        cbar_tick_size = 5000
        conv_plot = ax.imshow(conv_label, cmap='PiYG', interpolation='nearest', vmin=-cbar_tick_size,
                              vmax=cbar_tick_size)
        cbar = fig.colorbar(conv_plot, ticks=[-cbar_tick_size, 0, cbar_tick_size])
        cbar.ax.set_yticklabels(['< -{}'.format(cbar_tick_size), '0', '> {}'.format(cbar_tick_size)])
        if save:
            plt.savefig('plots/convolution/weight_conv_{}.png'.format(time.time()), dpi=300)
        else:
            plt.show()
        plt.close('all')


def calc_conv(connections, weight_matrices):
    weight_matrix = weight_matrices[connections[0]]
    if len(connections) == 1:
        return weight_matrix
    else:
        connections = connections[1:]
        conv_vec = []
        for n_id in range(weight_matrix.shape[0]):
            conv_vec.append(
                sum(map(lambda x: x[0] * x[1], zip(weight_matrix[n_id], calc_conv(connections, weight_matrices)))))
        return conv_vec


def plot_output_spike_count(output_spikes_per_label, plot_title, start_from, save=False, image_title=''):
    plt.clf()
    fig, ax = plt.subplots()
    size = output_spikes_per_label.shape[0]
    cax = ax.imshow(output_spikes_per_label, cmap='viridis', interpolation='nearest',
                    extent=[-0.5 + start_from, size - 0.5 + start_from, size - 0.5 + start_from, -0.5 + start_from], vmin=0)
    # ax.set_title(plot_title, y=1.08)
    ax.xaxis.tick_top()
    ax.set_xticks(range(start_from, size + start_from))
    ax.set_yticks(range(start_from, size + start_from))
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Actual label')
    ax.set_ylabel('Output neuron id')
    cbar = fig.colorbar(cax)
    if save:
        plt.savefig('plots/{}_{}.png'.format(image_title, time.time()), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')


def plot_attention_window_on_hist(df, win_x, win_y, attention_window_size, number, save=False):
    bucket = np.zeros((128, 128), dtype=int)
    for event in df.itertuples():
        bucket[event.y][event.x] += 1
    plt.clf()
    fig, ax = plt.subplots()
    rect_bucket = np.ones((128, 128), dtype=int)
    rect_bucket[np.clip(win_y, 0, 128):np.clip(win_y + attention_window_size, 0, 128),
    np.clip(win_x, 0, 128):np.clip(win_x + attention_window_size, 0, 128)] = 0
    rect_bucket[np.clip(win_y + 1, 0, 128):np.clip(win_y + attention_window_size - 1, 0, 128),
    np.clip(win_x + 1, 0, 128):np.clip(win_x + attention_window_size - 1, 0, 128)] = 1
    bucket *= rect_bucket
    rect_bucket -= 1
    bucket -= 10 * rect_bucket
    cax = ax.imshow(bucket, cmap='viridis', interpolation='nearest', vmin=0, vmax=10)
    ax.set_title('Attention window')
    # cbar = fig.colorbar(cax)
    if save:
        plt.savefig('plots/attention_window/att_win_{:05d}.png'.format(number), dpi=300)
    else:
        plt.show()
    plt.close('all')
