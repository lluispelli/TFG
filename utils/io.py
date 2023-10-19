import csv
import pdb
from os.path import join, dirname

import numpy as np
import torch
import PIL
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
color_code = ['b', 'r', 'k', 'g', 'm', 'y', 'c', 'pink']

def slices(slices_in,  # the 2D slices
           titles=None,  # list of titles
           cmaps=None,  # list of colormaps
           norms=None,  # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           grid=False,  # option to plot the images in a grid or a single row
           width=15,  # width in in
           show=True,  # option to actually show the plot (plt.show())
           axes_off=True,
           imshow_args=None):
    '''
    Plot a grid of slices (2d images)
    From neuron library
    Credit: https://github.com/adalca/neurite/blob/legacy/neuron/plot.py
    '''

    # input processing
    if isinstance(slices_in,torch.Tensor):
        slices_in = [slices_in.cpu().detach().numpy()]

    elif isinstance(slices_in, np.ndarray):
        slices_in = [slices_in]

    elif isinstance(slices_in, PIL.Image.Image):
        slices_in = [np.asarray(slices_in)]

    elif isinstance(slices_in, list):
        for it_s, s  in enumerate(slices_in):
            if isinstance(s, torch.Tensor):
                if s.is_cuda:
                    s = s.cpu()
                slices_in[it_s] = s.detach().numpy()

            elif isinstance(s, PIL.Image.Image):
                slices_in[it_s] = np.asarray(s)

    nb_plots = len(slices_in)
    for si, slice_in in enumerate(slices_in):
        if len(slice_in.shape) != 2:
            assert len(slice_in.shape) == 3 and slice_in.shape[-1] == 3, 'each slice has to be 2d or RGB (3 channels)'
        slices_in[si] = slice_in.astype('float')

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i], interpolation="nearest", norm=norms[i], **imshow_args[i])

        # colorbars
        # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        if do_colorbars and cmaps[i] is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_ax, cax=cax)

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        if axes_off:
            ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)

    if show:
        plt.tight_layout()
        plt.show()

    return (fig, axs)


def plot_results(train_file, val_file, keys=None, show=False, restrict_ylim=False):
    # past_backend = plt.get_backend()
    # plt.switch_backend('Agg')

    if keys is None: keys = ['Loss']

    iter_per_epoch, n_epochs, last_epoch = 0, 0, 0
    value_dict = {k: [] for k in keys}
    results_dict = {'Train': {k: {'x': [], 'y': []} for k in keys},
                    'Train_mean': {k: {'x': [], 'y': []} for k in keys},
                    'Train_median': {k: {'x': [], 'y': []} for k in keys},
                    'Validation': {k: {'x': [], 'y': []} for k in keys}}

    with open(train_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for it_row, row in enumerate(csvreader):
            for k in keys:
                if k in row:

                    iter_per_epoch = max(iter_per_epoch, int(row['Iteration']))
                    n_epochs = max(n_epochs, int(row['Epoch']))

                    results_dict['Train'][k]['x'] += [(int(row['Epoch']) - 1) * iter_per_epoch + int(row['Iteration'])]
                    results_dict['Train'][k]['y'] += [float(row[k])]

                    if int(row['Epoch']) > last_epoch:
                        last_epoch = int(row['Epoch'])
                        results_dict['Train_median'][k]['x'] += [int(row['Epoch'])*iter_per_epoch]
                        results_dict['Train_median'][k]['y'] += [float(np.median(value_dict[k]))]

                        results_dict['Train_mean'][k]['x'] += [int(row['Epoch'])*iter_per_epoch]
                        results_dict['Train_mean'][k]['y'] += [float(np.mean(value_dict[k]))]
                        value_dict[k] = [float(row[k])]

                    else:
                        value_dict[k] += [float(row[k])]


    with open(val_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for it_row, row in enumerate(csvreader):
            for k in keys:
                if k in row:
                    # print('warning. uncomment validation if new protocol is used')
                    results_dict['Validation'][k]['x'] += [int(row['Epoch'])*iter_per_epoch]
                    results_dict['Validation'][k]['y'] += [float(row[k])]
                    # results_dict['Validation'][k]['x'] += [int(row['Loss'])*iter_per_epoch]
                    # results_dict['Validation'][k]['y'] += [float(row[None][0])]

    x_ticks_label = np.linspace(0, n_epochs, 6).astype('int')
    x_ticks = np.linspace(0, iter_per_epoch*n_epochs, 6).astype('int')
    plt.figure()
    for k in keys:
        print(k)
        plt.plot(results_dict['Train'][k]['x'], results_dict['Train'][k]['y'], color='b', marker='*', alpha=0.6, label='Train iter.')
        plt.plot(results_dict['Train_median'][k]['x'], results_dict['Train_median'][k]['y'], color='r', marker='*', label='Median Ep. train')
        plt.plot(results_dict['Train_mean'][k]['x'], results_dict['Train_mean'][k]['y'], color='m', marker='*', label='Mean Ep. train')
        plt.plot(results_dict['Validation'][k]['x'], results_dict['Validation'][k]['y'], color='g', marker='*', label='Validation')

        if restrict_ylim:
            ymin, ymax = np.percentile(results_dict['Train'][k],1), np.percentile(results_dict['Train'][k],99)

            if ymin < 0:
                ymin = ymin*1.5
            else:
                ymin=ymin*0.5

            if ymax < 0:
                ymax = ymax*0.5
            else:
                ymax=ymax*1.5

            plt.ylim([ymin,ymax])

        plt.xticks(np.arange(0, n_epochs, 5*iter_per_epoch))

        plt.xticks(x_ticks, x_ticks_label)
        plt.xlabel('Epochs')
        plt.ylabel(k)
        plt.grid()
        ax = plt.gca()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol=4)
        if show:
            plt.show()
        else:
            plt.savefig(join(dirname(train_file), k + '_results.png'))

        plt.close()

    # plt.switch_backend(past_backend)

