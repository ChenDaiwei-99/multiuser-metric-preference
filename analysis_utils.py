# Utility files for data analysis and plotting

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 16}) 

import scipy.io as sio
import numpy as np
import dataset
import metrics
import utils
import scipy.spatial as sp
import math

plot_labels_color = {'frobM_l2v': 'Learned M, crowd',
                'I_l2v': 'Identity M',
                'frobM_l2v_single': 'Learned M, individual'}

plot_labels_normal = {'nucfull': 'Nuclear full',
                'nucM_l2v': 'Nuclear metric',
                'nucM_nucV': 'Nuclear split',
                'frobM_l2v': 'Frobenius metric',
                'psdM': 'PSD only',
                'nucfull_single': 'Nuclear full, single',
                'oracle': 'Oracle'}

def aggregate_color(root, idx_list):
    """
    Aggregates results file from color_experiment experiments

    Inputs:
        root: file path for .mat root (without indices)
        idx_list: list of indices to aggregate
    """

    results = {'methods': None,
               'N': None,
               'm_train_vec': None,
               'relative_M_error': None,
               'scaled_M_error': None,
               'test_accuracy': None,
               'M': None,
               'V': None
               }

    results_single = {
               'relative_M_error': None,
               'scaled_M_error': None,
               'test_accuracy': None,
               'M': None,
               'V': None
               }

    for idx in idx_list:
        results_idx = sio.loadmat(root + '_' + str(idx) + '.mat', squeeze_me=False)
        results_single_idx = sio.loadmat(root + '_' + str(idx) + '_single.mat', squeeze_me=False)

        if results['m_train_vec'] is None:
            for key in results_idx.keys():
                results[key] = results_idx[key]
        else:
            for key in ['relative_M_error', 'scaled_M_error',
                'test_accuracy', 'M', 'V']:

                results[key] = np.concatenate(
                    (results[key], results_idx[key]),
                    axis=1)

        if results_single['relative_M_error'] is None:
            for key in results_single_idx.keys():
                results_single[key] = results_single_idx[key]
        else:
            for key in ['relative_M_error', 'scaled_M_error',
                'test_accuracy', 'M', 'V']:

                results_single[key] = np.concatenate(
                    (results_single[key], results_single_idx[key]),
                    axis=1)

    sio.savemat(root + '.mat', results)
    sio.savemat(root + '_single.mat', results_single)


def aggregate_normal_noisy1bit(root, idx_list):
    """
    Aggregates results file from normal_noisy1bit experiments

    Inputs:
        root: file path for .mat root (without indices)
        idx_list: list of indices to aggregate
    """

    results = {'methods': None,
               'm_train_vec': None,
               'n': None,
               'd': None,
               'N' : None,
               'r_vec': None,
               'inherent_error': None,
               'relative_M_error': None,
               'relative_V_error': None,
               'relative_U_error': None,
               'scaled_M_error': None,
               'scaled_V_error': None,
               'test_accuracy': None,
               'M': None,
               'true_prediction_accuracy': None,
               'empirical_risk': None
               }

    for idx in idx_list:
        results_idx = sio.loadmat(root + '_' + str(idx) + '.mat', squeeze_me=False)

        if results['m_train_vec'] is None:
            for key in results_idx.keys():
                results[key] = results_idx[key]
        
        else:
            results['inherent_error'] = np.concatenate(
                (results['inherent_error'], results_idx['inherent_error']),
                axis=1)

            results['true_prediction_accuracy'] = np.concatenate(
                (results['true_prediction_accuracy'], results_idx['true_prediction_accuracy']),
                axis=1)
            
            for key in ['relative_M_error', 'relative_V_error',
                'relative_U_error', 'scaled_M_error', 'scaled_V_error',
                'test_accuracy', 'M', 'empirical_risk']:

                results[key] = np.concatenate(
                    (results[key], results_idx[key]),
                    axis=2)

    sio.savemat(root + '.mat', results)


def metrics_vs_npairs_color(results_root, ploterr=0, plotmed=False, fontsize=16,
                      figformat='png', figsize=(8,5), ylog=False, legend_order=None):
    """
    Generates figures plotting metrics against number of paired comparisons per user (from color experiments)

    Inputs:
        results_root: path to .mat file saved by run_experiments.py, without .mat extension
        ploterr: error type
            0: none
            1: standard deviation
            2: standard error
            3: 95-percentile
        plotmed: flag to plot median
        fontsize: axis label fontsize
        figformat: figure save format {'pdf', 'png'}
        figsize: figure size
        ylog: if True, plot y-axis on log scale
        legend_order: if not None, list of legend index order
    """
    
    # plotting setup
    percentile_alpha = 0.05
    col = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default color cycle
    linewidth = 1.75
    
    ylabels = {'relative_M_error': 'Relative metric error',
               'scaled_M_error': 'Scaled metric error',
               'test_accuracy': 'Test accuracy'}

    # load data
    results = sio.loadmat(results_root + '.mat')
    results_single = sio.loadmat(results_root + '_single.mat')

    methods = [met.strip() for met in list(results['methods'])]

    m_train_vec = results['m_train_vec'][0]

    for key in ['test_accuracy']:

        this_fig = plt.figure()
        plt.clf()

        single_aggregate = np.mean(results_single[key], axis=2)

        for mi in range(len(methods)):

            if mi < len(methods) - 1:
                this_data = results[key][:, :, mi]
            else:
                this_data = single_aggregate

            line = np.mean(this_data, axis=1)
            pline = plt.plot(m_train_vec, line, color=col[mi], linewidth=linewidth)

            err_std = np.std(this_data, axis=1)
            err_se =  err_std / np.sqrt(this_data.shape[1])

            if ploterr == 1:
                err_upper = line + err_std
                err_lower = line - err_std
            elif ploterr == 2:
                err_upper = line + err_se
                err_lower = line - err_se
            elif ploterr == 3:
                err_upper = np.percentile(a=this_data, q=100*(1-percentile_alpha/2), axis=1)
                err_lower = np.percentile(a=this_data, q=100*percentile_alpha/2, axis=1)

            if plotmed:
                med = np.median(this_data, axis=1)
                plt.plot(m_train_vec, med, color=col[mi], linestyle='--')

            if ploterr > 0:
                plt.fill_between(m_train_vec, err_lower, err_upper, color=col[mi], alpha=0.15)

            pline[0].set_label(plot_labels_color[methods[mi]])

        savepath = results_root + '_metrics_vs_npairs_color_{}'.format(key)
        make_figure(han=this_fig, xlabel='Number of pairs per user',
                    ylabel=ylabels[key], legend={}, legend_order=legend_order,
                    figsize=figsize, saveplotpath=savepath,
                    figformat=figformat, ylog=ylog, fontsize=fontsize)


def metrics_vs_npairs_normal(results_root, ploterr=0, plotmed=False,
                      figformat='png', figsize=(8,5), ylog=False,
                      enableTitle=True, fontsize=16):
    """
    Generates figures plotting metrics against number of paired comparisons per user (from normal_noisy1bit experiments)

    Inputs:
        results_root: path to .mat file saved by run_experiments.py, without .mat extension
        ploterr: error type
            0: none
            1: standard deviation
            2: standard error
            3: 95-percentile
        plotmed: flag to plot median
        figformat: figure save format {'pdf', 'png'}
        figsize: figure size
        ylog: if True, plot y-axis on log scale
        enableTrue: flag to enable title in figure
        fontsize: axis label font size
    """
    
    # plotting setup
    percentile_alpha = 0.05
    col = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default color cycle
    linewidth = 1.75
    subopt_thres = 0.95
    
    ylabels = {'relative_M_error': 'Metric (M) relative error',
               'relative_V_error': 'Pseudo-ideal point (v) relative error',
               'relative_U_error': 'Ideal point (u) relative error',
               'scaled_M_error': '||(M/||M||F - M*/||M*||F)||F',
               'scaled_V_error': '||(V/||V||F - V*/||V*||F)||F',
               'test_accuracy': 'Test accuracy'}

    # load data
    results = sio.loadmat(results_root + '.mat')

    r_vec = results['r_vec'].squeeze()
    n = results['n'].squeeze()
    d = results['d'].squeeze()
    N = results['N'].squeeze()
    m_train_vec = results['m_train_vec'].squeeze()
    methods = [met.strip() for met in list(results['methods'])]

    oracle_accuracy = results['true_prediction_accuracy']

    # sweep over metric ranks
    for ri in range(len(r_vec)):
        print('Analyzing rank {} / {}'.format(ri, len(r_vec)))

        r = r_vec[ri]

        for key in ['relative_M_error', 'relative_V_error',
                'relative_U_error', 'test_accuracy']:
            print('    Analyzing {}'.format(key))
            
            this_fig = plt.figure()
            plt.clf()

            if key == 'test_accuracy':
                methods_full = methods + ['oracle']
                legend_order = [0,2,1,3,4,5,6]
            else:
                methods_full = methods
                legend_order = [0,2,1,3]
            
            for mi in range(len(methods_full)):

                #if (methods_full[mi] == 'psdM') and key != 'test_accuracy':
                if (methods_full[mi] == 'psdM' or methods_full[mi] == 'nucfull_single') and key != 'test_accuracy':
                    continue

                if methods_full[mi] == 'oracle':
                    this_data = np.tile(oracle_accuracy[ri], (len(m_train_vec), 1))
                else:
                    this_data = results[key][ri, :, :, mi]

                line = np.mean(this_data, axis=1)
                pline = plt.plot(m_train_vec, line, color=col[mi], linewidth=linewidth)

                if key == 'test_accuracy' and methods_full[mi] == 'nucfull':
                    oracle_final = np.mean(oracle_accuracy[ri])
                    crossed_mask = line >= (oracle_final * subopt_thres)
                    if any(crossed_mask):
                        mi_crossed = np.argmax(crossed_mask)
                    else:
                        mi_crossed = -1
                    print('Nucfull crossed {:.2%} x oracle at measurement {} at rank {}'.format(
                            subopt_thres, m_train_vec[mi_crossed], r))


                err_std = np.std(this_data, axis=1)
                err_se =  err_std / np.sqrt(this_data.shape[1])

                if ploterr == 1:
                    err_upper = line + err_std
                    err_lower = line - err_std
                elif ploterr == 2:
                    err_upper = line + err_se
                    err_lower = line - err_se
                elif ploterr == 3:
                    err_upper = np.percentile(a=this_data, q=100*(1-percentile_alpha/2), axis=1)
                    err_lower = np.percentile(a=this_data, q=100*percentile_alpha/2, axis=1)

                if plotmed:
                    med = np.median(this_data, axis=1)
                    plt.plot(m_train_vec, med, color=col[mi], linestyle='--')

                if ploterr > 0:
                    plt.fill_between(m_train_vec, err_lower, err_upper, color=col[mi], alpha=0.15)

                pline[0].set_label(plot_labels_normal[methods_full[mi]])

            if enableTitle:
                title = 'd:{}, r:{}, n:{}, N:{}'.format(d, r, n, N)
            else:
                title=None

            savepath = results_root + '_metrics_vs_npairs_d{}_r{}_{}'.format(d, r, key)
            make_figure(han=this_fig, xlabel='Number of pairs per user',
                        ylabel=ylabels[key], title=title,
                        legend={'loc':'best'}, fontsize=fontsize,
                        figsize=figsize, saveplotpath=savepath,
                        figformat=figformat, ylog=ylog, legend_order=legend_order)

def display_metric_color(results_root):
    """
    Displays aggregate metric and standard error of entries
    """
    
    # load data
    results = sio.loadmat(results_root + '.mat')
    methods = [met.strip() for met in list(results['methods'])]
    frobM_idx = methods.index('frobM_l2v')
    assert frobM_idx == 0 # sanity check

    M_allT = results['M'][-1, :, frobM_idx, :, :]
    M_scaledT = np.zeros(M_allT.shape)
    for ti in range(M_scaledT.shape[0]):
        M_scaledT[ti] = M_allT[ti] / np.linalg.norm(M_allT[ti], 'fro')

    M_all_mean = np.mean(M_allT, axis=0)
    M_all_se = np.std(M_allT, axis=0) / np.sqrt(M_allT.shape[0])

    M_scaled_mean = np.mean(M_scaledT, axis=0)
    M_scaled_se = np.std(M_scaledT, axis=0) / np.sqrt(M_scaledT.shape[0])

    output_file = results_root + '_Mcrowd.txt'
    output_mat = results_root + '_Mcrowd.mat'

    strout = 'Ave(M):\n'
    for ri in range(M_all_mean.shape[0]):
        for ci in range(M_all_mean.shape[1]):
            strout += '{:>10.3f}, '.format(M_all_mean[ri,ci])
        strout += '\n'
    strout += '\n'

    strout += 'StandErr(M):\n'
    for ri in range(M_all_se.shape[0]):
        for ci in range(M_all_se.shape[1]):
            strout += '{:>10.3f}, '.format(M_all_se[ri,ci])
        strout += '\n'
    strout += '\n'

    strout += 'Ave(Mscaled):\n'
    for ri in range(M_scaled_mean.shape[0]):
        for ci in range(M_scaled_mean.shape[1]):
            strout += '{:>10.3f}, '.format(M_scaled_mean[ri,ci])
        strout += '\n'
    strout += '\n'

    strout += 'StandErr(Mscaled):\n'
    for ri in range(M_scaled_se.shape[0]):
        for ci in range(M_scaled_se.shape[1]):
            strout += '{:>10.3f}, '.format(M_scaled_se[ri,ci])
        strout += '\n'
    strout += '\n'

    with open(output_file, 'w') as file:
        file.write(strout)

    sio.savemat(output_mat, {'M_all_mean': M_all_mean, 'M_all_se': M_all_se,
                             'M_scaled_mean': M_scaled_mean, 'M_scaled_se': M_scaled_se})

def make_figure(han, xlabel='', ylabel='', title=None, legend=None, legend_order=None, figsize=(8,5),
                saveplotpath=None, xr=None, yr=None, ylog=False, figformat='png', fontsize=16):
    """
    Formats and saves figure in input figure handle.
    
    Inputs:
        han: figure handle (object)
        xlabel: x label
        ylabel: y label
        legend: dictionary of legend arguments. If None, legend not called
        legend_order: if not None, list of legend index order
        figsize: figure size
        saveplotpath: file path for saving figures, root. (None for no saving)
        xr: xlim range (default None)
        yr: ylim range (default None)
        ylog: if True, plot y-axis on log scale
        figformat: figure save format {'pdf', 'png'}
        fontsize: axis label font size
    Outputs:
        han: figure handle
    """
    
    plt.figure(han.number)
    
    # handled globally
    #matplotlib.rcParams['pdf.fonttype'] = 42
    #matplotlib.rcParams['ps.fonttype'] = 42
    #plt.rcParams.update({'font.size': fontsize}) 

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)       

    if title is not None:
        plt.title(title)    
        
    if xr:
        plt.xlim(xr)
        
    if yr:
        plt.ylim(yr)
        
    if ylog:
        plt.yscale('log')
            
    if legend is not None:
        leg = plt.legend(**legend)

        if legend_order is not None:
            leg_handles, leg_labels = plt.gca().get_legend_handles_labels()
            leg = plt.legend([leg_handles[idx] for idx in legend_order],
                [leg_labels[idx] for idx in legend_order])

        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.5)
    
    plt.grid(which='major',linewidth='0.5')
    han.set_size_inches(figsize)
    
    if saveplotpath:
        plt.savefig(fname=saveplotpath + '.{}'.format(figformat), format=figformat, bbox_inches='tight')
    
    return han

if __name__ == '__main__':

    colorexps = 1
    normal_highnoise = 1
    normal_medNoise = 1

    if colorexps:

        # color experiments
        root = '../../results/final/color_experiment_N48_165264247294/color_experiment_N48_165264247294'
        idx_list = list(range(30))

        aggregate_color(root, idx_list)
        
        plt.close('all')
        metrics_vs_npairs_color(root, ploterr=3, plotmed=False,
            figformat='pdf', ylog=False, legend_order=[1,0,2], fontsize=16)

        display_metric_color(root)

        root_lowQ = '../../results/final/color_experiment_N48_lowQ_165264809219/color_experiment_N48_lowQ_165264809219'
        idx_list = list(range(30))

        aggregate_color(root_lowQ, idx_list)

        plt.close('all')
        metrics_vs_npairs_color(root_lowQ, ploterr=3, plotmed=False,
            figformat='pdf', ylog=False, legend_order=[1,0,2])

    if normal_highnoise:

        # normal experiments - high noise
        root = '../../results/final/normal_noisy1bit_highNoise_165262847637/normal_noisy1bit_highNoise_165262847637'
        idx_list = list(range(30))
        
        aggregate_normal_noisy1bit(root, idx_list)

        metrics_vs_npairs_normal(root, enableTitle=False,
            ploterr=2, plotmed=False, figformat='pdf', ylog=False)
    
    if normal_medNoise:
        # normal experiments - medium noise
        root = '../../results/final/normal_noisy1bit_medNoise_165270786371/normal_noisy1bit_medNoise_165270786371'
        idx_list = list(range(30))
        
        aggregate_normal_noisy1bit(root, idx_list)

        metrics_vs_npairs_normal(root, enableTitle=False,
            ploterr=2, plotmed=False, figformat='pdf', ylog=False)
