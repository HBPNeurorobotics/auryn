import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import file_io as fio
import jaer_data_handler as jhandler
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import glob
import os


class Plotter:
    def __init__(self, path_to_pkg, path_to_plots=""):
        self.path_to_pkg = path_to_pkg
        if path_to_plots:
            self.path_to_plots = path_to_plots
        else:
            self.path_to_plots = '{}/scripts/plots'.format(self.path_to_pkg)
        try:
            os.makedirs(self.path_to_plots)
        except OSError as e:
            print(e)

    def plot_2d_input_ras(self, path, dimension, start=0, end=sys.maxint, save=False):
        if start > end:
            print('start time has to be smaller than end time.')
            return
        res = dimension * dimension
        stepsize = 3000
        bucket = np.zeros((dimension, dimension), dtype=int)
        labels = np.array([])
        seek = 0
        for i in xrange(end // stepsize + 1):
            local_start = start + i * stepsize
            local_end = (i + 1) * stepsize
            if end < local_end:
                local_end = end
            df, seek = fio.ras_to_df("{}/scripts/inputs/{}/input.ras".format(self.path_to_pkg, path), local_start,
                                     local_end,
                                     seek)
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
        self.plot_heat_map(bucket, 'Spike count from {}s to {}s (labels: {})'.format(start, end, labels),
                           image_title='{}'.format(time.time()), save=save, dynamic_v=True)

    def plot_2d_from_txt(self, path, title, start=0, end=9223372036854775807):
        bucket = np.loadtxt(
            '{path_to_plots}/{path}_ras_input_{start}_to_{end}.txt'.format(path_to_plots=self.path_to_plots, path=path,
                                                                           start=start, end=end))
        self.plot_heat_map(bucket, title)

    def plot_2d_hist_from_aedat31(self, pathname, start=0, end=sys.maxint, image_title=''):
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
        self.plot_heat_map(bucket, 'Spatial event distribution - label 1'.format(start, end), save=True,
                           image_title=image_title, dynamic_v=True)

    def plot_2d_events_from_df(self, df, centroid=None, plot_title='', image_title='', hist_shape=(128, 128),
                               legend=True):
        bucket = np.zeros(hist_shape, dtype=int)
        for event in df.itertuples():
            if event.p == 1:
                bucket[event.y][event.x] += 1
            elif event.p == 0:
                bucket[event.y][event.x] -= 1
        bucket[bucket > 0] = 1
        bucket[bucket < 0] = -1
        self.plot_heat_map(bucket, plot_title, save=True,
                           image_title=image_title, show_cbar=False, vmin=-1, vmax=1,
                           centroid=centroid, legend=legend)

    def plot_2d_events_from_aedat(self, path, start=0, end=sys.maxint, image_title='', version='aedat3',
                                  attention_window=False, event_amount=1000):
        if version == 'aedat3':
            timestamps, xaddr, yaddr, pol = jhandler.load_aedat31(path, debug=0)
        else:
            timestamps, xaddr, yaddr, pol = jhandler.load_jaer(path, version='aedat', debug=0)
            timestamps = np.array(timestamps)
            if timestamps[0] > timestamps[-1]:
                print('HAD TO RESTORE TS ORDER')
                timestamps = self.restore_ts_order(timestamps)
            timestamps -= min(timestamps)
        df = pd.DataFrame({'ts': timestamps, 'x': xaddr, 'y': yaddr, 'p': pol})
        df.ts = df.ts * 1e-6
        if end > max(df.ts):
            end = max(df.ts)
        df = df[(df.ts >= start) & (df.ts <= end)]
        if attention_window:
            centroid = df.loc[:, ['x', 'y']].rolling(window=event_amount, min_periods=1).median().astype(int).mean()
        else:
            centroid = None
        plot_title = 'Events from {:0.2f}s to {:.2f}s'.format(start, end)
        self.plot_2d_events_from_df(df, centroid, plot_title, image_title)

    def restore_ts_order(self, timestamps):
        for i in range(len(timestamps) - 1):
            if timestamps[i] > timestamps[i + 1]:
                timestamps[:i + 1] -= (2 ** 32 * 1e-6)
                return timestamps

    def plot_heat_map(self, bucket, plot_title, save=False, image_title='', show_cbar=True, vmin=0, vmax=10,
                      dynamic_v=False, centroid=None, attention_window_size=64, legend=True):
        plt.clf()
        fig = plt.figure(frameon=False, figsize=(5, 5))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        if dynamic_v:
            cax = ax.imshow(bucket, cmap='viridis', interpolation='nearest', aspect='auto')
        else:
            cax = ax.imshow(bucket, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax, aspect='auto')
        # ax.set_title(plot_title)
        if show_cbar:
            cbar = fig.colorbar(cax)
        else:
            custom_lines = [Line2D([0], [0], color=plt.cm.viridis(1.), lw=0, marker='s', markersize=20),
                            Line2D([0], [0], color=plt.cm.viridis(-1.), lw=0, marker='s', markersize=20)]
            if legend:
                plt.legend(custom_lines, ['ON', 'OFF'], handlelength=1, borderpad=0.5, framealpha=0.5, numpoints=1, prop={'size': 20})

        # fig.tight_layout()
        if centroid is not None:
            rect = patches.Rectangle((int(centroid.centroid_x - attention_window_size / 2.),
                                      int(centroid.centroid_y - attention_window_size / 2.)),
                                     attention_window_size, attention_window_size,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.autoscale(False)
        extent = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())

        if save:
            plt.savefig('{}/{}.png'.format(self.path_to_plots, image_title), dpi=300, bbox_inches=extent)
        else:
            plt.show()
        plt.close('all')

    def plot_weight_matrix(self, path, connections=['vh', 'hh', 'ho'], save=False):
        num_subplots = len(connections)
        for i, connection in enumerate(connections):
            weight_matrix = fio.mtx_file_to_matrix(path.format(connection))
            plt.subplot(num_subplots, 1, i + 1)
            plt.gca().set_title(connection)
            cax = plt.imshow(weight_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
            cbar = plt.colorbar(cax)
        plt.tight_layout()
        if save:
            plt.savefig('{}/weight_matrix_{}_{}.png'.format(self.path_to_plots, connections, time.time()), dpi=300)
        else:
            plt.show()
        plt.close('all')

    def plot_output_spikes_aggregated(self, path, start, end, classes, save=False, output_path=''):
        data_df, seek = fio.ras_to_df(path, start, end)
        for i, c in enumerate(classes):
            class_df = data_df.loc[data_df.n_id == i]
            plt.plot([0] + list(class_df.ts), [0] + list(class_df['n_id'].expanding().count()), label=c)
        plt.legend(classes)
        plt.tight_layout()
        # plt.title("Aggregated output spikes")
        plt.xlabel("Time in seconds")
        plt.ylabel("Number of spikes")
        if save:
            if not output_path:
                out_path = self.path_to_plots
                name = 'agg_output_spikes_{}.png'.format(time.time())
            else:
                out_path = output_path
                name = 'agg_output_spikes.png'
            plt.savefig('{}/{}'.format(out_path, name), dpi=300)
        else:
            plt.show()
        plt.clf()

    def plot_ras_spikes(self, pathinput, start, end, layers=['vis', 'hid', 'out'], res=sys.maxint, number_of_classes=10,
                        save=False, input_att_window=False, att_win_input_size=128 * 2, output_path='',
                        plot_label=True):
        title = 'Spike times'
        counter = 1
        num_plots = len(layers)
        if 'vis' in layers:
            if plot_label:
                num_plots += 1
            if input_att_window:
                num_plots += 1

        for i, layer in enumerate(layers):
            path = pathinput.format(layer)
            data_df, seek = fio.ras_to_df(path, start, end)
            ax1 = plt.subplot(num_plots, 1, counter)
            if counter == 1:
                pass
                # ax1.set_title(title)
            ax1.get_yaxis().set_label_coords(-0.1, 0.5)
            if layer == 'vis':
                label_df = data_df.loc[data_df.n_id >= res]
                data_df = data_df.loc[data_df.n_id < res]
                if plot_label:
                    for i in xrange(number_of_classes):
                        ax1.plot((start, end), (i, i), 'r--', linewidth=0.1, alpha=1)
                    ax1.plot(label_df.ts.values, label_df.n_id.values - res, linestyle='None', marker=u'|',
                             color=[0, 0, 1, 1],
                             markersize=1, alpha=0.1)
                    ax1.set_ylabel("Label")
                    ax1.xaxis.set_major_locator(plt.NullLocator())
                    counter += 1
                    x1, x2, y1, y2 = ax1.axis()
                    ax1.axis((start, end, y1, y2))
                if input_att_window:
                    data_neuron_size = res - att_win_input_size
                    att_pos_df = data_df.loc[data_df.n_id >= data_neuron_size]
                    data_df = data_df.loc[data_df.n_id < data_neuron_size]
                    ax1 = plt.subplot(num_plots, 1, counter, sharex=ax1)
                    ax1.get_yaxis().set_label_coords(-0.1, 0.5)
                    ax1.plot(att_pos_df.ts.values, att_pos_df.n_id.values, linestyle='None', marker=u',',
                             color=[0, 0, 1, 1])
                    ax1.set_ylabel("Attention")
                    ax1.xaxis.set_major_locator(plt.NullLocator())
                    counter += 1
                    x1, x2, y1, y2 = ax1.axis()
                    ax1.axis((start, end, y1, y2))
                ax1 = plt.subplot(num_plots, 1, counter)
                ax1.get_yaxis().set_label_coords(-0.1, 0.5)
                ax1.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                         markersize=1, alpha=0.4)
                ax1.set_ylabel("Input")
                ax1.xaxis.set_major_locator(plt.NullLocator())
            elif layer == 'hid':
                ax1.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                         markersize=1, alpha=0.4)
                ax1.plot((start, end), (200, 200), 'r--', linewidth=0.5, alpha=1)
                ax1.set_ylabel("Hidden")

                if i != len(layers) - 1:
                    ax1.xaxis.set_major_locator(plt.NullLocator())
            elif layer == 'out':
                for i in xrange(number_of_classes):
                    ax1.plot((start, end), (i, i), 'r--', linewidth=0.1, alpha=1)
                ax1.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                         markersize=1, alpha=0.2)
                ax1.set_ylabel("Output")
            elif layer == 'err1':
                for i in xrange(number_of_classes):
                    ax1.plot((start, end), (i, i), 'r--', linewidth=0.1, alpha=1)
                ax1.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                         markersize=1, alpha=0.2)
                path = pathinput.format(layer.replace('1', '2'))
                data_df, seek = fio.ras_to_df(path, start, end)
                ax1.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u'|', color=[1, 0, 0, 1],
                         markersize=1, alpha=0.2)
                ax1.set_ylabel("Error")
                ax1.xaxis.set_major_locator(plt.NullLocator())
            # print(data_df.ts.values.size)
            counter += 1
            x1, x2, y1, y2 = ax1.axis()
            ax1.axis((start, end, y1, y2))

        ax1.set_xlabel("Time in seconds")
        plt.tight_layout()

        if save:
            if not output_path:
                out_path = self.path_to_plots
                name = '{}_{}_{}_{}.png'.format(start, end, layers, time.time())
            else:
                out_path = output_path
                name = 'spiketrain.png'
            plt.savefig('{}/{}'.format(out_path, name), dpi=300)
        else:
            plt.show()
        plt.clf()

    def plot_weight_stats(self, stats, save=False):
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
            plt.savefig('{}/weight_stats_{}.png'.format(self.path_to_plots, time.time()), dpi=300)
        else:
            plt.show()
        plt.close('all')

    def plot_output_weights_over_time(self, output_weights, save=False):
        plt.clf()
        plt.plot(output_weights, alpha=0.1)
        if save:
            plt.savefig('{}/output_weights_{}.png'.format(self.path_to_plots, time.time()), dpi=300)
        else:
            plt.show()
        plt.close('all')

    def plot_accuracy_rate_first(self, acc_hist, save=False):
        # x = [i[0] for i in acc_hist]
        y_rate = [i[1][0] for i in acc_hist]
        y_first = [i[1][1] for i in acc_hist]
        plt.plot(y_rate, marker='o')
        plt.plot(y_first, marker='o', color='r')
        # plt.title('classification accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy in percent')
        plt.legend(['rate', 'first'])
        if save:
            plt.savefig('{}/acc_hist_{}.png'.format(self.path_to_plots, time.time()), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close('all')

    def plot_weight_histogram(self, path, nh1, connections=['vh', 'hh', 'ho'], save=False):
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
            plt.savefig('{}/weight_histogram_{}_{}.png'.format(self.path_to_plots, connections, time.time()), dpi=300)
        else:
            plt.show()
        plt.close('all')

    def plot_confusion_matrix(self, df_confusion, save=False):
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
            plt.savefig('{}/confusion_matrix_{}.png'.format(self.path_to_plots, time.time()), dpi=300,
                        bbox_inches='tight')
        else:
            plt.show()
        plt.close('all')

    def plot_weight_convolution(self, path, nh1, nc, connections=['vh', 'hh', 'ho'], save=False,
                                cbar_tick_size=5000, labels=[]):
        weight_matrices = {}
        for connection in connections:
            weight_matrix = fio.mtx_file_to_matrix(
                "{path}/fwmat_{connection}.mtx".format(path=path, connection=connection))
            if connection == 'vh':
                weight_matrix = weight_matrix[:32 * 32]
            elif connection == 'hh':
                weight_matrix = weight_matrix[:, nh1:]
            elif connection == 'ho':
                weight_matrix = weight_matrix[nh1:]
            weight_matrices[connection] = weight_matrix
        conv_matrix = self.calc_conv(connections, weight_matrices)  # .reshape(32,32,12)
        # conv_matrix = np.array(map(lambda x: np.argmax(x), conv_matrix)).reshape(32,32)
        os.makedirs('{}/convolution'.format(self.path_to_plots))
        for i in range(nc):
            plt.clf()
            conv_label = np.array([item[i] for item in conv_matrix]).reshape(32, 32)
            fig, ax = plt.subplots()
            if labels:
                ax.set_title('Weight convolution for label {}'.format(labels[i]))
            else:
                ax.set_title('Weight convolution for label {}'.format(i))
            conv_plot = ax.imshow(conv_label, cmap='PiYG', interpolation='nearest', vmin=-cbar_tick_size,
                                  vmax=cbar_tick_size)
            cbar = fig.colorbar(conv_plot, ticks=[-cbar_tick_size, 0, cbar_tick_size])
            cbar.ax.set_yticklabels(['< -{}'.format(cbar_tick_size), '0', '> {}'.format(cbar_tick_size)])
            if save:
                plt.savefig('{}/convolution/weight_conv_{}.png'.format(self.path_to_plots, time.time()), dpi=300)
            else:
                plt.show()
            plt.close('all')

    def calc_conv(self, connections, weight_matrices):
        weight_matrix = weight_matrices[connections[0]]
        if len(connections) == 1:
            return weight_matrix
        else:
            connections = connections[1:]
            conv_vec = []
            for n_id in range(weight_matrix.shape[0]):
                conv_vec.append(
                    sum(map(lambda x: x[0] * x[1],
                            zip(weight_matrix[n_id], self.calc_conv(connections, weight_matrices)))))
            return conv_vec

    def plot_output_spike_count(self, output_spikes_per_label, plot_title, start_from, save=False, image_title=''):
        plt.clf()
        fig, ax = plt.subplots()
        size = output_spikes_per_label.shape[0]
        cax = ax.imshow(output_spikes_per_label, cmap='viridis', interpolation='nearest',
                        extent=[-0.5 + start_from, size - 0.5 + start_from, size - 0.5 + start_from, -0.5 + start_from],
                        vmin=0)
        # ax.set_title(plot_title, y=1.08)
        ax.xaxis.tick_top()
        ax.set_xticks(range(start_from, size + start_from))
        ax.set_yticks(range(start_from, size + start_from))
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('Actual label')
        ax.set_ylabel('Output neuron id')
        cbar = fig.colorbar(cax)
        if save:
            plt.savefig('{}/{}_{}.png'.format(self.path_to_plots, image_title, time.time()), dpi=300,
                        bbox_inches='tight')
        else:
            plt.show()
        plt.close('all')

    def plot_attention_window_on_hist(self, df, win_x, win_y, attention_window_size, number, save=False):
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
            plt.savefig('{}/attention_window/att_win_{:05d}.png'.format(self.path_to_plots, number), dpi=300)
        else:
            plt.show()
        plt.close('all')
