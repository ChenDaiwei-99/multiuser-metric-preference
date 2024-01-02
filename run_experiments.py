# Script to run experiments

import yaml
import dataset
import numpy as np
import math
import multipref
import metrics
import scipy.io as sio
import logging
import pickle
import cvxpy as cp
import utils

cvxkwargs_global_normal = {'solver':cp.SCS, 'eps':1e-6}
cvxkwargs_global_color = {'solver':cp.CVXOPT}

def normal_noisy1bit(pm, methods=['nucfull', 'nucM_l2v', 'nucM_nucV',
    'frobM_l2v', 'psdM', 'nucfull_single'], save_root=None, loglevel=None, seed=None):
    """
    pm: parameter dictionary
    methods: list of methods to evaluate (must be in the following order: ['nucfull', 'nucM_l2v', 'nucM_nucV',
        'frobM_l2v', 'psdM', 'nucfull_single'])
    save_root: path to save file (without extensions)
    loglevel: level for logging, if applicable
    seed: numpy random number generator seed (in [0, 2**32)). None for no seeding
    """

    # check that all keys are specified
    expected_keys = ['T', 'd', 'rstart', 'rstop', 'rstep',
                     'n', 'N', 'noise_type', 'noise_param',
                     'train_frac', 'n_splits', 'loss_fun', 'm_train',
                     'hyper_sweep', 'use_oracle', 'm_train_res']

    for key in expected_keys:
        assert key in pm.keys(), '{} missing from pm'.format(key)
    
    # 1 / pm['train_frac'] needs to be integer for proper indexing
    assert 1 / pm['train_frac'] == round(1 / pm['train_frac'])

    # save parameters
    if save_root is not None:
        with open(save_root + '.pkl', 'wb') as handle:
            pickle.dump({'pm':pm, 'seed':seed}, handle)

    # set seeding
    if seed is not None:
        np.random.seed(seed)

    # hyperparameters to sweep, if applicable
    hyper_sweep = np.array(pm['hyper_sweep'])

    # misc initialization
    d = pm['d']
    D = int(d*(d+1)/2)
    n = pm['n']
    
    logflag = (save_root is not None) and (loglevel is not None)

    if logflag:
        logging.basicConfig(filename=save_root+'.log', level=loglevel,
            format='%(process)d - %(asctime)s - %(message)s')
        
        logging.info('Experiment saving to {}'.format(save_root))

    if pm['rstart'] is None:
        r_vec = np.array([d])
    elif pm['rstop'] is None:
        r_vec = np.arange(start=pm['rstart'], stop=d + 1, step=pm['rstep'], dtype=int)
    else:
        r_vec = np.arange(start=pm['rstart'], stop=pm['rstop'] + 1, step=pm['rstep'], dtype=int)

    # trim r_vec
    r_vec = r_vec[r_vec <= d]

    m_train = pm['m_train']
    m_train_res = pm['m_train_res']
    if m_train_res is None:
        m_train_vec = np.array(m_train)
    else:
        m_train_vec = np.arange(start=m_train_res, stop=m_train + 1, step=m_train_res)
    m_all = int(m_train / pm['train_frac'])

    # initialize results data
    results = {'methods': methods,
               'm_train_vec': m_train_vec,
               'n': n,
               'd': d,
               'N': pm['N'],
               'r_vec': r_vec,
               'inherent_error': np.zeros((len(r_vec), pm['T'])),
               'relative_M_error': np.zeros((len(r_vec), len(m_train_vec), pm['T'], len(methods))),
               'relative_V_error': np.zeros((len(r_vec), len(m_train_vec), pm['T'], len(methods))),
               'relative_U_error': np.zeros((len(r_vec), len(m_train_vec), pm['T'], len(methods))),
               'scaled_M_error': np.zeros((len(r_vec), len(m_train_vec), pm['T'], len(methods))),
               'scaled_V_error': np.zeros((len(r_vec), len(m_train_vec), pm['T'], len(methods))),
               'test_accuracy': np.zeros((len(r_vec), len(m_train_vec), pm['T'], len(methods))),
               'M': np.zeros((len(r_vec), len(m_train_vec), pm['T'], len(methods), d, d)),
               'true_prediction_accuracy': np.zeros((len(r_vec), pm['T'])),
               'empirical_risk': np.zeros((len(r_vec), len(m_train_vec), pm['T'], len(methods)))
               }

    for ti in range(pm['T']):

        if logflag:
            logging.info('Trial {} / {}'.format(ti+1, pm['T']))

        # generate items, to be used across M of all ranks in this trial. N=1 is dummy
        normal_dataset = dataset.Dataset(dataset_type='Normal',
                    d=d, r=d, n=n, N=1, m=m_all, noise_type=pm['noise_type'],
                    noise_param=pm['noise_param'], X=None)

        data = normal_dataset.getAllData()
        X = data['X']

        for ri in range(len(r_vec)):

            if logflag:
                logging.info('    Rank {} / {}'.format(ri+1, len(r_vec)))
            
            r = r_vec[ri]

            if pm['N'] is None:
                N = r
            else:
                N = pm['N']

            # keep same items, but generate new metric and comparisons
            normal_dataset = dataset.Dataset(dataset_type='Normal',
                    d=d, r=r, n=n, N=N, m=m_all, noise_type=pm['noise_type'],
                    noise_param=pm['noise_param'], X=X)

            data = normal_dataset.getAllData()
            Mtrue = data['M']
            Utrue = data['U']
            Vtrue = -2 * Mtrue @ Utrue

            results['inherent_error'][ri, ti] = np.mean(data['Y'] != normal_dataset.Y_noiseless)

            S_train, S_test, Y_train, Y_test = normal_dataset.getTrainTestSplit(train_size = pm['train_frac'])

            # true prediction accuracy
            Y_pred_true = utils.predict(X, S_test, Mtrue, Vtrue)
            results['true_prediction_accuracy'][ri,ti] = np.mean(Y_pred_true == Y_test)

            method_idx = -1
            if 'nucfull' in methods:
                method_idx += 1

                if logflag:
                    logging.info('        Method: nucfull')

                ########## nucfull
                model_nucfull = multipref.MultiPref(d=d, N=N, method='nucfull')

                # set hyperparameter
                if pm['use_oracle']:
                    lams_nucfull = [np.linalg.norm(np.hstack((Mtrue, Vtrue)), 'nuc')] # oracle hyperparameter
                else:
                    lams_nucfull = [np.sqrt(r*(d**2 + d*N))] # approximation from expectation

                # fit model
                model_nucfull.fit(X=X, S=S_train, Y=Y_train, lams=lams_nucfull,
                                verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                noise_param=pm['noise_param'], use_logger=logflag,
                                projPSD=True, cvxkwargs=cvxkwargs_global_normal)

                # record full data metrics
                metrics_output = normal_noisy1bit_metrics(Mtrue = Mtrue, Utrue = Utrue, model = model_nucfull,
                    X = X, S_test = S_test, Y_test = Y_test)

                for key in ['relative_M_error', 'relative_V_error', 'relative_U_error',
                    'test_accuracy', 'scaled_M_error', 'scaled_V_error',
                    'empirical_risk']:
                    results[key][ri,-1,ti,method_idx] = metrics_output[key]

                results['M'][ri,-1,ti,method_idx,:,:] = model_nucfull.M

            if 'nucM_l2v' in methods:
                method_idx += 1

                if logflag:
                    logging.info('        Method: nucM_l2v')

                ########## nucM, l2v
                model_nucM_l2v = multipref.MultiPref(d=d, N=N, method='nucM_l2v')

                # set hyperparameter
                if pm['use_oracle']:
                    lams_nucM_l2v = [np.linalg.norm(Mtrue, 'nuc'),
                        max([np.linalg.norm(v) for v in Vtrue.T])] # oracle hyperparameter
                else:
                    lams_nucM_l2v = [np.sqrt(r)*d, np.sqrt(d)] # approximation from expectation
                
                # fit model
                model_nucM_l2v.fit(X=X, S=S_train, Y=Y_train, lams=lams_nucM_l2v,
                                verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                noise_param=pm['noise_param'], use_logger=logflag,
                                projPSD=True, cvxkwargs=cvxkwargs_global_normal)

                # record full data metrics
                metrics_output = normal_noisy1bit_metrics(Mtrue = Mtrue, Utrue = Utrue,
                    model = model_nucM_l2v, X = X, S_test = S_test, Y_test = Y_test)
                
                for key in ['relative_M_error', 'relative_V_error', 'relative_U_error',
                    'test_accuracy', 'scaled_M_error', 'scaled_V_error',
                    'empirical_risk']:
                    results[key][ri,-1,ti,method_idx] = metrics_output[key]

                results['M'][ri,-1,ti,method_idx,:,:] = model_nucM_l2v.M

            if 'nucM_nucV' in methods:
                method_idx += 1

                if logflag:
                    logging.info('        Method: nucM_nucV')

                ########## nucM, l2v
                model_nucM_nucV = multipref.MultiPref(d=d, N=N, method='nucM_nucV')

                # set hyperparameter
                if pm['use_oracle']:
                    lams_nucM_nucV = [np.linalg.norm(Mtrue, 'nuc'), np.linalg.norm(Vtrue, 'nuc')] # oracle hyperparameter
                else:
                    lams_nucM_nucV = [np.sqrt(r)*d, np.sqrt(r*N*d)] # approximation from expectation
                
                # fit model
                model_nucM_nucV.fit(X=X, S=S_train, Y=Y_train, lams=lams_nucM_nucV,
                                    verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                    noise_param=pm['noise_param'], use_logger=logflag,
                                    projPSD=True, cvxkwargs=cvxkwargs_global_normal)

                # record full data metrics
                metrics_output = normal_noisy1bit_metrics(Mtrue = Mtrue, Utrue = Utrue,
                    model = model_nucM_nucV, X = X, S_test = S_test, Y_test = Y_test)
                
                for key in ['relative_M_error', 'relative_V_error', 'relative_U_error',
                    'test_accuracy', 'scaled_M_error', 'scaled_V_error',
                    'empirical_risk']:
                    results[key][ri,-1,ti,method_idx] = metrics_output[key]

                results['M'][ri,-1,ti,method_idx,:,:] = model_nucM_nucV.M
            
            if 'frobM_l2v' in methods:
                method_idx += 1

                if logflag:
                    logging.info('        Method: frobM_l2v')

                ########## frobM, l2v
                model_frobM_l2v = multipref.MultiPref(d=d, N=N, method='frobM_l2v')

                # set hyperparameter
                if pm['use_oracle']:
                    lams_frobM_l2v = [np.linalg.norm(Mtrue, 'fro'),
                        max([np.linalg.norm(v) for v in Vtrue.T])] # oracle hyperparameter
                else:
                    lams_frobM_l2v = [d, np.sqrt(d)] # approximation from expectation
                
                # fit model
                model_frobM_l2v.fit(X=X, S=S_train, Y=Y_train, lams=lams_frobM_l2v,
                                    verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                    noise_param=pm['noise_param'], use_logger=logflag,
                                    projPSD=True, cvxkwargs=cvxkwargs_global_normal)

                # record full data metrics
                metrics_output = normal_noisy1bit_metrics(Mtrue = Mtrue, Utrue = Utrue,
                    model = model_frobM_l2v, X = X, S_test = S_test, Y_test = Y_test)
                for key in ['relative_M_error', 'relative_V_error', 'relative_U_error',
                    'test_accuracy', 'scaled_M_error', 'scaled_V_error',
                    'empirical_risk']:
                    results[key][ri,-1,ti,method_idx] = metrics_output[key]

                results['M'][ri,-1,ti,method_idx,:,:] = model_frobM_l2v.M

            if 'psdM' in methods:
                method_idx += 1

                if logflag:
                    logging.info('        Method: psdM')

                ########## psdM
                model_psdM = multipref.MultiPref(d=d, N=N, method='psdM')
                
                # fit model
                model_psdM.fit(X=X, S=S_train, Y=Y_train, lams=[],
                                    verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                    noise_param=pm['noise_param'], use_logger=logflag,
                                    projPSD=True, cvxkwargs=cvxkwargs_global_normal)

                # record full data metrics
                metrics_output = normal_noisy1bit_metrics(Mtrue = Mtrue, Utrue = Utrue,
                    model = model_psdM, X = X, S_test = S_test, Y_test = Y_test)
                for key in ['relative_M_error', 'relative_V_error', 'relative_U_error',
                    'test_accuracy', 'scaled_M_error', 'scaled_V_error',
                    'empirical_risk']:
                    results[key][ri,-1,ti,method_idx] = metrics_output[key]

                results['M'][ri,-1,ti,method_idx,:,:] = model_psdM.M

            
            # organize training measurements by user
            S_train_sets = []
            Y_train_sets = []
            S_test_sets = []
            Y_test_sets = []

            pre_sweep_idx = method_idx
            for k in range(N):
                S_train_k, Y_train_k = zip(*[(s, y) for (s, y) in zip(S_train, Y_train) if s[0] == k])
                S_train_sets.append(list(S_train_k))
                Y_train_sets.append(list(Y_train_k))

                S_test_k, Y_test_k = zip(*[(s, y) for (s, y) in zip(S_test, Y_test) if s[0] == k])
                S_test_sets.append(list(S_test_k))
                Y_test_sets.append(list(Y_test_k))

                # reindex user locally
                S_train_k = [(0, s[1]) for s in S_train_k]
                S_test_k = [(0, s[1]) for s in S_test_k]
                
                method_idx = pre_sweep_idx
                if 'nucfull_single' in methods:
                    method_idx += 1
                    
                    if logflag:
                        logging.info('        Method: nucfull_single, user {} / {}'.format(k+1, N))

                    ########## nucfull single (per user)

                    model_nucfull_k = multipref.MultiPref(d=d, N=1, method='nucfull')

                    # set hyperparameter
                    if pm['use_oracle']:
                        lams_nucfull_k = [np.linalg.norm(
                            np.hstack((Mtrue, Vtrue[:, k, None])), 'nuc')] # oracle hyperparameter
                    else:
                        lams_nucfull_k = [np.sqrt(r*(d**2 + d))] # approximation from expectation

                    # fit model
                    model_nucfull_k.fit(X=X, S=S_train_k, Y=Y_train_k, lams=lams_nucfull_k,
                                    verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                    noise_param=pm['noise_param'], use_logger=logflag,
                                    projPSD=True, cvxkwargs=cvxkwargs_global_normal)
                    params_k = model_nucfull_k.get_params()

                    # record metrics
                    metrics_output = normal_noisy1bit_metrics(Mtrue = Mtrue, Utrue = Utrue[:, k, None],
                    model = model_nucfull_k, X = X, S_test = S_test_k, Y_test = Y_test_sets[k])

                    for key in ['relative_M_error', 'relative_V_error', 'relative_U_error',
                    'test_accuracy', 'scaled_M_error', 'scaled_V_error',
                    'empirical_risk']:
                        results[key][ri,-1,ti,method_idx] += (1 / N) * metrics_output[key]

            ################################################################
            # compute progressive metrics

            if logflag:
                logging.info('        Computing progressive trials across all methods...')

            # progress through partial measurements
            for mi in range(len(m_train_vec) - 1):
                m_train_i = m_train_vec[mi]

                if logflag:
                    logging.info('            Measurement {} / {}'.format(m_train_i, m_train_vec[-1]))

                # accumulate training points up to m_train_i
                S_train_i = sum([S_train_k[:m_train_i] for S_train_k in S_train_sets], [])
                Y_train_i = sum([Y_train_k[:m_train_i] for Y_train_k in Y_train_sets], [])

                method_idx = -1
                if 'nucfull' in methods:
                    method_idx += 1

                    if logflag:
                        logging.info('            Method: nucfull')

                    ####### nucfull
                    model_nucfull = multipref.MultiPref(d=d, N=N, method='nucfull')

                    # fit model
                    model_nucfull.fit(X=X, S=S_train_i, Y=Y_train_i, lams=lams_nucfull, verbose=False,
                                    sample_size=None, loss_fun=pm['loss_fun'],
                                    noise_param=pm['noise_param'], use_logger=logflag,
                                    projPSD=True, cvxkwargs=cvxkwargs_global_normal)

                    # record full data metrics
                    metrics_output = normal_noisy1bit_metrics(Mtrue = Mtrue, Utrue = Utrue, model = model_nucfull,
                        X = X, S_test = S_test, Y_test = Y_test)
                    
                    for key in ['relative_M_error', 'relative_V_error', 'relative_U_error',
                    'test_accuracy', 'scaled_M_error', 'scaled_V_error',
                    'empirical_risk']:
                        results[key][ri,mi,ti,method_idx] = metrics_output[key]

                    results['M'][ri,mi,ti,method_idx,:,:] = model_nucfull.M


                if 'nucM_l2v' in methods:
                    method_idx += 1

                    if logflag:
                        logging.info('            Method: nucM_l2v')

                    ####### nucM, l2v
                    model_nucM_l2v = multipref.MultiPref(d=d, N=N, method='nucM_l2v')
                    
                    # fit model
                    model_nucM_l2v.fit(X=X, S=S_train_i, Y=Y_train_i, lams=lams_nucM_l2v, verbose=False,
                                    sample_size=None, loss_fun=pm['loss_fun'],
                                    noise_param=pm['noise_param'], use_logger=logflag,
                                    projPSD=True, cvxkwargs=cvxkwargs_global_normal)

                    # record full data metrics
                    metrics_output = normal_noisy1bit_metrics(Mtrue = Mtrue, Utrue = Utrue,
                        model = model_nucM_l2v, X = X, S_test = S_test, Y_test = Y_test)

                    for key in ['relative_M_error', 'relative_V_error', 'relative_U_error',
                    'test_accuracy', 'scaled_M_error', 'scaled_V_error',
                    'empirical_risk']:
                        results[key][ri,mi,ti,method_idx] = metrics_output[key]

                    results['M'][ri,mi,ti,method_idx,:,:] = model_nucM_l2v.M

                if 'nucM_nucV' in methods:
                    method_idx += 1

                    if logflag:
                        logging.info('            Method: nucM_nucV')

                    ####### nucM, l2v
                    model_nucM_nucV = multipref.MultiPref(d=d, N=N, method='nucM_nucV')
                    
                    # fit model
                    model_nucM_nucV.fit(X=X, S=S_train_i, Y=Y_train_i, lams=lams_nucM_nucV, verbose=False,
                                        sample_size=None, loss_fun=pm['loss_fun'],
                                        noise_param=pm['noise_param'], use_logger=logflag,
                                        projPSD=True, cvxkwargs=cvxkwargs_global_normal)

                    # record full data metrics
                    metrics_output = normal_noisy1bit_metrics(Mtrue = Mtrue, Utrue = Utrue,
                        model = model_nucM_nucV, X = X, S_test = S_test, Y_test = Y_test)
                    
                    for key in ['relative_M_error', 'relative_V_error', 'relative_U_error',
                    'test_accuracy', 'scaled_M_error', 'scaled_V_error',
                    'empirical_risk']:
                        results[key][ri,mi,ti,method_idx] = metrics_output[key]

                    results['M'][ri,mi,ti,method_idx,:,:] = model_nucM_nucV.M
                
                if 'frobM_l2v' in methods:
                    method_idx += 1

                    if logflag:
                        logging.info('            Method: frobM_l2v')

                    ####### frobM, l2v
                    model_frobM_l2v = multipref.MultiPref(d=d, N=N, method='frobM_l2v')
                    
                    # fit model
                    model_frobM_l2v.fit(X=X, S=S_train_i, Y=Y_train_i, lams=lams_frobM_l2v, verbose=False,
                                        sample_size=None, loss_fun=pm['loss_fun'],
                                        noise_param=pm['noise_param'], use_logger=logflag,
                                        projPSD=True, cvxkwargs=cvxkwargs_global_normal)

                    # record full data metrics
                    metrics_output = normal_noisy1bit_metrics(Mtrue = Mtrue, Utrue = Utrue,
                        model = model_frobM_l2v, X = X, S_test = S_test, Y_test = Y_test)

                    for key in ['relative_M_error', 'relative_V_error', 'relative_U_error',
                    'test_accuracy', 'scaled_M_error', 'scaled_V_error',
                    'empirical_risk']:
                        results[key][ri,mi,ti,method_idx] = metrics_output[key]

                    results['M'][ri,mi,ti,method_idx,:,:] = model_frobM_l2v.M

                if 'psdM' in methods:
                    method_idx += 1

                    if logflag:
                        logging.info('            Method: psdM')

                    ####### psdM
                    model_psdM = multipref.MultiPref(d=d, N=N, method='psdM')
                    
                    # fit model
                    model_psdM.fit(X=X, S=S_train_i, Y=Y_train_i, lams=[], verbose=False,
                                        sample_size=None, loss_fun=pm['loss_fun'],
                                        noise_param=pm['noise_param'], use_logger=logflag,
                                        projPSD=True, cvxkwargs=cvxkwargs_global_normal)

                    # record full data metrics
                    metrics_output = normal_noisy1bit_metrics(Mtrue = Mtrue, Utrue = Utrue,
                        model = model_psdM, X = X, S_test = S_test, Y_test = Y_test)

                    for key in ['relative_M_error', 'relative_V_error', 'relative_U_error',
                    'test_accuracy', 'scaled_M_error', 'scaled_V_error',
                    'empirical_risk']:
                        results[key][ri,mi,ti,method_idx] = metrics_output[key]

                    results['M'][ri,mi,ti,method_idx,:,:] = model_psdM.M

                if 'nucfull_single' in methods:
                    method_idx += 1

                    ########## nucfull single (per user)
                    for k in range(N):

                        if logflag:
                            logging.info('            Method: nucfull_single, user {} / {}'.format(k+1, N))

                        # accumulate training points up to m_train_i
                        S_train_k = S_train_sets[k][:m_train_i]
                        Y_train_k = Y_train_sets[k][:m_train_i]

                        # reindex user locally
                        S_train_k = [(0, s[1]) for s in S_train_k]
                        S_test_k = [(0, s[1]) for s in S_test_sets[k]]
                        
                        model_nucfull_k = multipref.MultiPref(d=d, N=1, method='nucfull')

                        # set hyperparameter
                        if pm['use_oracle']:
                            lams_nucfull_k = [np.linalg.norm(
                                np.hstack((Mtrue, Vtrue[:, k, None])), 'nuc')] # oracle hyperparameter
                        else:
                            lams_nucfull_k = [np.sqrt(r*(d**2 + d))] # approximation from expectation

                        # fit model
                        model_nucfull_k.fit(X=X, S=S_train_k, Y=Y_train_k, lams=lams_nucfull_k,
                                        verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                        noise_param=pm['noise_param'], use_logger=logflag,
                                        projPSD=True, cvxkwargs=cvxkwargs_global_normal)
                        params_k = model_nucfull_k.get_params()

                        # record metrics
                        metrics_output = normal_noisy1bit_metrics(Mtrue = Mtrue, Utrue = Utrue[:, k, None],
                        model = model_nucfull_k, X = X, S_test = S_test_k, Y_test = Y_test_sets[k])

                        for key in ['relative_M_error', 'relative_V_error', 'relative_U_error',
                        'test_accuracy', 'scaled_M_error', 'scaled_V_error',
                        'empirical_risk']:
                            results[key][ri,mi,ti,method_idx] += (1 / N) * metrics_output[key]

                if save_root is not None:
                    sio.savemat(save_root + '.mat', results) # save after every iteration
        

def normal_noisy1bit_metrics(Mtrue, Utrue, model, X, S_test, Y_test):
    params = model.get_params()
    Mhat, Vhat = params['M'], params['V']
    Uhat = model.estimate_users(Q=model.d)
    
    return {
        'relative_M_error': metrics.relative_M_error(Mtrue=Mtrue, Mhat=Mhat),
        'relative_V_error': metrics.relative_V_error(Mtrue=Mtrue, Utrue=Utrue, Vhat=Vhat),
        'relative_U_error': metrics.relative_U_error(Mtrue=Mtrue, Utrue=Utrue, Uhat=Uhat),
        'test_accuracy': metrics.prediction_accuracy(model=model, X=X, S=S_test, Y=Y_test),
        'scaled_M_error': metrics.scaled_M_error(Mtrue=Mtrue, Mhat=Mhat),
        'scaled_V_error': metrics.scaled_V_error(Mtrue=Mtrue, Utrue=Utrue, Vhat=Vhat),
        'empirical_risk': model.getEmpiricalRisk()
    }


def color_experiment(pm, methods=['frobM_l2v', 'I_l2v', 'frobM_l2v_single'], save_root=None, loglevel=None, seed=None):
    """
    pm: parameter dictionary
    methods: list of methods to evaluate (must be in the following order: ['frobM_l2v', 'I_l2v', 'frobM_l2v_single'])
    save_root: path to save file (without extensions)
    loglevel: level for logging, if applicable
    seed: numpy random number generator seed (in [0, 2**32)). None for no seeding
    """

    # check that all keys are specified
    expected_keys = ['T', 'N', 'm_train', 'n_splits', 'loss_fun',
                     'm_train_res', 'hyper_sweep']

    for key in expected_keys:
        assert key in pm.keys(), '{} missing from pm'.format(key)

    # save parameters
    if save_root is not None:
        with open(save_root + '.pkl', 'wb') as handle:
            pickle.dump({'pm':pm, 'seed':seed}, handle)

    # set seeding
    if seed is not None:
        np.random.seed(seed)

    # hyperparameters to sweep, if applicable
    hyper_sweep = np.array(pm['hyper_sweep'])

    # misc initialization
    N = pm['N']

    logflag = (save_root is not None) and (loglevel is not None)

    if logflag:
        logging.basicConfig(filename=save_root+'.log', level=loglevel,
            format='%(process)d - %(asctime)s - %(message)s')
        
        logging.info('Experiment saving to {}'.format(save_root))

    m_train = pm['m_train']
    m_train_res = pm['m_train_res']
    if m_train_res is None:
        m_train_vec = np.array(m_train)
    else:
        m_train_vec = np.arange(start=m_train_res, stop=m_train + 1, step=m_train_res)

    # initialize results data
    if 'frobM_l2v_single' in methods:
        n_grouped_methods = len(methods)-1  # FIXME: why?
    else:
        n_grouped_methods = len(methods)

    results = {'methods': methods,
               'N': N,
               'm_train_vec': m_train_vec,
               'relative_M_error': np.zeros((len(m_train_vec), pm['T'], n_grouped_methods)),
               'scaled_M_error': np.zeros((len(m_train_vec), pm['T'], n_grouped_methods)),
               'test_accuracy': np.zeros((len(m_train_vec), pm['T'], n_grouped_methods)),
               'test_accuracy_individual': np.zeros((len(m_train_vec), pm['T'], N, n_grouped_methods)),
               'M': np.zeros((len(m_train_vec), pm['T'], n_grouped_methods, 3, 3)),
               'V': np.zeros((len(m_train_vec), pm['T'], n_grouped_methods, 3, N))
               }

    results_single = {'relative_M_error': np.zeros((len(m_train_vec), pm['T'], N)),
                      'scaled_M_error': np.zeros((len(m_train_vec), pm['T'], N)),
                      'test_accuracy': np.zeros((len(m_train_vec), pm['T'], N)),
                      'M': np.zeros((len(m_train_vec), pm['T'], N, 3, 3)),
                      'V': np.zeros((len(m_train_vec), pm['T'], N, 3))
                      }

    for ti in range(pm['T']):

        if logflag:
            logging.info('Trial {} / {}'.format(ti+1, pm['T']))

        # load color data
        # color_dataset = dataset.Dataset(dataset_type='Color',
        #     color_path='../data/CPdata.mat', N=N)
        # FIXME: change the color_path by daiwei
        color_dataset = dataset.Dataset(dataset_type='Color',
            color_path='./CPdata.mat', N=N)

        data = color_dataset.getAllData()
        X = data['X']
        max_x_norm = max(np.linalg.norm(X, axis=0))

        Mtrue = np.eye(3) # approximation to what we expect from CIELAB space

        S_train, S_test, Y_train, Y_test = color_dataset.getTrainTestSplit(
            train_size = m_train * N)
        
        # FIXME: substitute these methods by pytorch version
        method_idx = -1
        if 'frobM_l2v' in methods:
            method_idx += 1

            if logflag:
                logging.info('    Method: frobM_l2v')

            ########## frobM, l2v
            model_frobM_l2v = multipref.MultiPref(d=3, N=N, method='frobM_l2v')

            # set hyperparameter
            lams_frobM_l2v = [np.sqrt(3), 2* max_x_norm]
            # ||I||F = sqrt(3), which is the ground-truth M we expect
            # ||v|| = ||-2 M u|| = 2 ||I u|| = 2 ||u|| <~ 2 max_x ||x||_2 if u ~ x
            
            # fit model
            model_frobM_l2v.fit(X=X, S=S_train, Y=Y_train, lams=lams_frobM_l2v,
                                verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                cvxkwargs=cvxkwargs_global_color, use_logger=logflag, projPSD=True)

            # record full data metrics
            metrics_output = color_metrics(Mtrue=Mtrue, model=model_frobM_l2v, X=X, S=S_test, Y=Y_test)
            
            for key in ['relative_M_error', 'scaled_M_error', 'test_accuracy']:
                results[key][-1,ti,method_idx] = metrics_output[key]

            params = model_frobM_l2v.get_params()
            results['M'][-1,ti,method_idx,:,:] = params['M']
            results['V'][-1,ti,method_idx,:,:] = params['V']

        if 'I_l2v' in methods:
            method_idx += 1

            if logflag:
                logging.info('    Method: I_l2v')

            ########## I_l2v
            model_I_l2v = multipref.MultiPref(d=3, N=N, method='frobM_l2v')

            # set hyperparameter
            lams_I_l2v = [np.sqrt(3), 2* max_x_norm]
            # ||I||F = sqrt(3), which is the ground-truth M we expect
            # ||v|| = ||-2 M u|| = 2 ||I u|| = 2 ||u|| <~ 2 max_x ||x||_2 if u ~ x
            
            # fit model
            model_I_l2v.fit(X=X, S=S_train, Y=Y_train, lams=lams_I_l2v,
                                verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                cvxkwargs=cvxkwargs_global_color, Mfixed=Mtrue,
                                use_logger=logflag, projPSD=True)

            # record full data metrics
            metrics_output = color_metrics(Mtrue=Mtrue, model=model_I_l2v, X=X, S=S_test, Y=Y_test)

            for key in ['relative_M_error', 'scaled_M_error', 'test_accuracy']:
                results[key][-1,ti,method_idx] = metrics_output[key]

            params = model_I_l2v.get_params()
            results['M'][-1,ti,method_idx,:,:] = params['M']
            results['V'][-1,ti,method_idx,:,:] = params['V']

        ########## per user methods
        
        # organize training measurements by user
        S_train_sets = []
        Y_train_sets = []
        S_test_sets = []
        Y_test_sets = []

        for k in range(N):
            S_train_k, Y_train_k = zip(*[(s, y) for (s, y) in zip(S_train, Y_train) if s[0] == k])
            S_train_sets.append(list(S_train_k))
            Y_train_sets.append(list(Y_train_k))

            S_test_k, Y_test_k = zip(*[(s, y) for (s, y) in zip(S_test, Y_test) if s[0] == k])
            S_test_sets.append(list(S_test_k))
            Y_test_sets.append(list(Y_test_k))

            # individual evaluations on group methods
            method_idx = -1
            if 'frobM_l2v' in methods:
                method_idx += 1

                results['test_accuracy_individual'][-1,ti,k,method_idx] = \
                    metrics.prediction_accuracy(model=model_frobM_l2v, X=X, S=S_test_k, Y=Y_test_k)

            if 'I_l2v' in methods:
                method_idx += 1

                results['test_accuracy_individual'][-1,ti,k,method_idx] = \
                    metrics.prediction_accuracy(model=model_I_l2v, X=X, S=S_test_k, Y=Y_test_k)

            # reindex user locally
            S_train_k = [(0, s[1]) for s in S_train_k]
            S_test_k = [(0, s[1]) for s in S_test_k]

            if 'frobM_l2v_single' in methods:

                if logflag:
                    logging.info('    Method: frobM_l2v_single, user {} / {}'.format(k+1, N))

                ########## frobM, l2v
                model_frobM_l2v_k = multipref.MultiPref(d=3, N=1, method='frobM_l2v')
                
                # fit model
                model_frobM_l2v_k.fit(X=X, S=S_train_k, Y=Y_train_k, lams=lams_frobM_l2v,
                                    verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                    cvxkwargs=cvxkwargs_global_color, use_logger=logflag, projPSD=True)

                params_k = model_frobM_l2v_k.get_params()

                # record full data metrics
                metrics_output = color_metrics(Mtrue=Mtrue, model=model_frobM_l2v_k, X=X, S=S_test_k, Y=Y_test_k)

                for key in ['relative_M_error', 'scaled_M_error', 'test_accuracy']:
                    results_single[key][-1,ti,k] = metrics_output[key]

                results_single['M'][-1,ti,k,:,:] = params_k['M']
                results_single['V'][-1,ti,k,:] = params_k['V'].squeeze()

        ################################################################
        # compute progressive metrics

        if logflag:
            logging.info('    Computing progressive trials across all methods...')

        # progress through partial measurements
        for mi in range(len(m_train_vec) - 1):
            m_train_i = m_train_vec[mi]

            if logflag:
                logging.info('        Measurement {} / {}'.format(m_train_i, m_train_vec[-1]))

            # accumulate training points up to m_train_i
            S_train_i = sum([S_train_k[:m_train_i] for S_train_k in S_train_sets], [])
            Y_train_i = sum([Y_train_k[:m_train_i] for Y_train_k in Y_train_sets], [])

            method_idx = -1
            if 'frobM_l2v' in methods:
                method_idx += 1

                if logflag:
                    logging.info('            Method: frobM_l2v')

                ########## frobM, l2v
                model_frobM_l2v = multipref.MultiPref(d=3, N=N, method='frobM_l2v')
                
                # fit model
                model_frobM_l2v.fit(X=X, S=S_train_i, Y=Y_train_i, lams=lams_frobM_l2v,
                                    verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                    cvxkwargs=cvxkwargs_global_color, use_logger=logflag, projPSD=True)

                # record full data metrics
                metrics_output = color_metrics(Mtrue=Mtrue, model=model_frobM_l2v, X=X, S=S_test, Y=Y_test)

                for key in ['relative_M_error', 'scaled_M_error', 'test_accuracy']:
                    results[key][mi,ti,method_idx] = metrics_output[key]

                params = model_frobM_l2v.get_params()
                results['M'][mi,ti,method_idx,:,:] = params['M']
                results['V'][mi,ti,method_idx,:,:] = params['V']

            if 'I_l2v' in methods:
                method_idx += 1

                if logflag:
                    logging.info('            Method: I_l2v')

                ########## I_l2v
                model_I_l2v = multipref.MultiPref(d=3, N=N, method='frobM_l2v')
                
                # fit model
                model_I_l2v.fit(X=X, S=S_train_i, Y=Y_train_i, lams=lams_I_l2v,
                                    verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                    cvxkwargs=cvxkwargs_global_color, Mfixed=Mtrue,
                                    use_logger=logflag, projPSD=True)

                # record full data metrics
                metrics_output = color_metrics(Mtrue=Mtrue, model=model_I_l2v, X=X, S=S_test, Y=Y_test)

                for key in ['relative_M_error', 'scaled_M_error', 'test_accuracy']:
                    results[key][mi,ti,method_idx] = metrics_output[key]

                params = model_I_l2v.get_params()
                results['M'][mi,ti,method_idx,:,:] = params['M']
                results['V'][mi,ti,method_idx,:,:] = params['V']
            
            for k in range(N):

                # accumulate training points up to m_train_i
                S_train_k = S_train_sets[k][:m_train_i]
                Y_train_k = Y_train_sets[k][:m_train_i]
                Y_test_k = Y_test_sets[k]

                # individual evaluations on group methods
                method_idx = -1
                if 'frobM_l2v' in methods:
                    method_idx += 1

                    results['test_accuracy_individual'][mi,ti,k,method_idx] = \
                        metrics.prediction_accuracy(model=model_frobM_l2v, X=X, S=S_test_sets[k], Y=Y_test_k)

                if 'I_l2v' in methods:
                    method_idx += 1

                    results['test_accuracy_individual'][mi,ti,k,method_idx] = \
                        metrics.prediction_accuracy(model=model_I_l2v, X=X, S=S_test_sets[k], Y=Y_test_k)

                # reindex user locally
                S_train_k = [(0, s[1]) for s in S_train_k]
                S_test_k = [(0, s[1]) for s in S_test_sets[k]]
                
                if 'frobM_l2v_single' in methods:

                    if logflag:
                        logging.info('            Method: frobM_l2v, user {} / {}'.format(k+1, N))

                    ########## frobM, l2v
                    model_frobM_l2v_k = multipref.MultiPref(d=3, N=1, method='frobM_l2v')
                    
                    # fit model
                    model_frobM_l2v_k.fit(X=X, S=S_train_k, Y=Y_train_k, lams=lams_frobM_l2v,
                                        verbose=False, sample_size=None, loss_fun=pm['loss_fun'],
                                        cvxkwargs=cvxkwargs_global_color, use_logger=logflag, projPSD=True)

                    params_k = model_frobM_l2v_k.get_params()

                    # record full data metrics
                    metrics_output = color_metrics(Mtrue=Mtrue, model=model_frobM_l2v_k, X=X, S=S_test_k, Y=Y_test_k)

                    for key in ['relative_M_error', 'scaled_M_error', 'test_accuracy']:
                        results_single[key][mi,ti,k] = metrics_output[key]

                    results_single['M'][mi,ti,k,:,:] = params_k['M']
                    results_single['V'][mi,ti,k,:] = params_k['V'].squeeze()

            # save after every iteration
            if save_root is not None:
                sio.savemat(save_root + '.mat', results)
                sio.savemat(save_root + '_single.mat', results_single) 


def color_metrics(Mtrue, model, X, S, Y):
    params = model.get_params()
    Mhat, Vhat = params['M'], params['V']
    
    return {
        'relative_M_error': metrics.relative_M_error(Mtrue=Mtrue, Mhat=Mhat),
        'scaled_M_error': metrics.scaled_M_error(Mtrue=Mtrue, Mhat=Mhat),
        'test_accuracy': metrics.prediction_accuracy(model=model, X=X, S=S, Y=Y)
    }


def run_experiment(pm, methods, save_root=None, seed=None):
    # pm: experiment parameters
    # methods: list of methods to evaluate (must be in code order)
    # save_root: path root for file saving
    # seed: numpy random number generator seed (in [0, 2**32)). None for no seeding

    # experiment functions dictionary
    exp_funs = {'normal_noisy1bit': normal_noisy1bit,
                'color_experiment': color_experiment}

    assert pm['fun'] in exp_funs.keys()

    exp_fun = exp_funs[pm['fun']]
    exp_fun(pm=pm, methods=methods, save_root=save_root, loglevel=logging.INFO, seed=seed)


if __name__ == '__main__':

    # example experiment call
    with open('./configs/config_normal_medNoise.yaml') as f:
        config = yaml.safe_load(f)

    run_experiment(pm=config['pm'], methods=config['methods'], save_root='./config_normal_rsweep_lowNoise', seed=10)
