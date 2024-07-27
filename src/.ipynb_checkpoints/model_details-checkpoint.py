import os
import copy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn import metrics
import scipy
import matplotlib.pyplot as plt

import psytrack as psy
from psytrack.helper.crossValidation import split_data
from psytrack.helper.crossValidation import xval_loglike

def timing_sigmoid(x,params,min_val = -1, max_val = 0,tol=1e-3):
    '''
        Evaluates a sigmoid between min_val and max_val with parameters params
    '''
    if np.isnan(x):
        x = 0 
    y = min_val+(max_val-min_val)/(1+(x/params[1])**params[0])
    if (y -min_val) < tol:
        y = min_val
    if (max_val - y) < tol:
        y = max_val
    return y

def fit_weights(psydata, strategies, fit_overnight=False):
    '''
        does weight and hyper-parameter optimization on the data in psydata
        Args: 
            psydata is a dictionary with key/values:
            psydata['y'] = a vector of no-licks (1) and licks(2) for each images
            psydata['inputs'] = a dictionary with each key an input 
                ('random','timing', 'task', etc) each value has a 2D array of 
                shape (N,M), where N is number of imagees, and M is 1 unless 
                you want to look at history/image interaction terms

        RETURNS:
        hyp
        evd
        wMode
        hess
    '''
    # Set up number of regressors
    weights = {}
    for strat in strategies:
        weights[strat] = 1
    print(weights)
    K = np.sum([weights[i] for i in weights.keys()])

    # Set up initial hyperparameters
    hyper = {'sigInit': 2**4.,
            'sigma':[2**-4.]*K,
            'sigDay': 2**4}

    # Only used if we are fitting multiple sessions
    # where we have a different prior
    if fit_overnight:
        optList=['sigma','sigDay']
    else:
        optList=['sigma']
    
    # Do the fit
    hyp,evd,wMode,hess = psy.hyperOpt(psydata,hyper,weights, optList)
    credibleInt = hess['W_std']
    
    return hyp, evd, wMode, hess, credibleInt, weights

def transform(series):
    '''
        passes the series through the logistic function
    '''
    return 1/(1+np.exp(-(series)))


def compute_ypred(psydata, wMode, weights):
    '''
        Makes a full model prediction from the wMode
        Returns:
        pR, the probability of licking on each image
        pR_each, the contribution of licking from each weight. These contributions 
            interact nonlinearly, so this is an approximation. 
    '''
    g = psy.read_input(psydata, weights)
    gw = g*wMode.T
    total_gw = np.sum(gw,axis=1)
    pR = transform(total_gw)
    pR_each = transform(gw) 
    return pR, pR_each

def get_clean_string(strings):
    '''
        Return a cleaned up list of weights suitable for plotting labels
    '''
    string_dict = {
        'bias':'licking bias',
        'omissions':'omission',
        'omissions0':'omission',
        'Omissions':'omission',
        'Omissions1':'post omission',
        'omissions1':'post omission',
        'task0':'visual',
        'Task0':'visual',
        'timing1D':'timing',
        'Full-Task0':'full model',
        'dropout_task0':'Visual Dropout',    
        'dropout_timing1D':'Timing Dropout', 
        'dropout_omissions':'Omission Dropout',
        'dropout_omissions1':'Post Omission Dropout',
        'Sst-IRES-Cre' :'Sst Inhibitory',
        'Vip-IRES-Cre' :'Vip Inhibitory',
        'Slc17a7-IRES2-Cre' :'Excitatory',
        'strategy_dropout_index': 'strategy index',
        'num_hits':'rewards/session',
        'num_miss':'misses/session',
        'num_image_false_alarm':'false alarms/session',
        'num_post_omission_licks':'post omission licks/session',
        'num_omission_licks':'omission licks/session',
        'post_reward':'previous bout rewarded',
        'not post_reward':'previous bout unrewarded',
        'timing1':'1',
        'timing2':'2',
        'timing3':'3',
        'timing4':'4',
        'timing5':'5',
        'timing6':'6',
        'timing7':'7',
        'timing8':'8',
        'timing9':'9',
        'timing10':'10',    
        'not visual_strategy_session':'timing sessions',
        'visual_strategy_session':'visual sessions',
        'visual_only_dropout_index':'visual index',
        'timing_only_dropout_index':'timing index',
        'lick_hit_fraction_rate':'lick hit fraction',
        'session_roc':'dynamic model (AUC)',
        'miss':'misses',
        }

    clean_strings = []
    for w in strings:
        if w in string_dict.keys():
            clean_strings.append(string_dict[w])
        else:
            clean_strings.append(str(w).replace('_',' '))
    return clean_strings

def get_weights_list(weights): 
    '''
        Return a sorted list of the weights in the model
    '''
    weights_list = []
    for i in sorted(weights.keys()):
        weights_list += [i]*weights[i]
    return weights_list

def compute_cross_validation(psydata, hyp, weights,folds=10):
    '''
        Computes Cross Validation for the data given the regressors as 
        defined in hyp and weights
    '''
    trainDs, testDs = split_data(psydata,F=folds)
    test_results = []
    for k in range(folds):
        print("\rrunning fold " +str(k),end="") 
        _,_,wMode_K,_ = psy.hyperOpt(trainDs[k], hyp, weights, ['sigma'],hess_calc=None)
        logli, gw = xval_loglike(testDs[k], wMode_K, trainDs[k]['missing_trials'], 
            weights)
        res = {'logli' : np.sum(logli), 'gw' : gw, 'test_inds' : testDs[k]['test_inds']}
        test_results += [res]
   
    print("") 
    return test_results


def compute_cross_validation_ypred(psydata,test_results,ypred):
    '''
        Computes the predicted outputs from cross validation results by stitching 
        together the predictions from each folds test set

        full_pred is a vector of probabilities (0,1) for each time bin in psydata
    '''
    # combine each folds predictions
    myrange = np.arange(0, len(psydata['y']))
    xval_mask = np.ones(len(myrange)).astype(bool)
    X = np.array([i['gw'] for i in test_results]).flatten()
    test_inds = np.array([i['test_inds'] for i in test_results]).flatten()
    inrange = np.where((test_inds >= 0) & (test_inds < len(psydata['y'])))[0]
    inds = [i for i in np.argsort(test_inds) if i in inrange]
    X = X[inds]
    cv_pred = 1/(1+np.exp(-X))

    # Fill in untested indicies with ypred, these come from end
    full_pred = copy.copy(ypred)
    full_pred[np.where(xval_mask==True)[0]] = cv_pred
    return full_pred
 
 
def dropout_analysis(psydata, strategies,format_options):
    '''
        Computes a dropout analysis for the data in psydata. 
        In general, computes a full set, and then removes each feature one by one. 

        Returns a list of models and a list of labels for each dropout
    '''
    models =dict()

    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,strategies)
    cross_psydata = psy.trim(psydata, 
        END=int(np.floor(len(psydata['y'])/format_options['num_cv_folds'])\
        *format_options['num_cv_folds'])) 
    cross_results = compute_cross_validation(cross_psydata, hyp, weights,
        folds=format_options['num_cv_folds'])
    models['Full'] = (hyp, evd, wMode, hess, credibleInt,weights,cross_results)

    # Iterate through strategies and remove them
    for s in strategies:
        dropout_strategies = copy.copy(strategies)
        dropout_strategies.remove(s)
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,
            dropout_strategies)
        cross_results = compute_cross_validation(cross_psydata, hyp, weights,
            folds=format_options['num_cv_folds'])
        models[s] = (hyp, evd, wMode, hess, credibleInt,weights,cross_results)

    return models


def compute_model_roc(fit,plot_this=False,cross_validation=True):
    '''
        Computes area under the ROC curve for the model in fit. 
        
        plot_this (bool), plots the ROC curve. 
        cross_validation (bool)
            if True uses the cross validated prediction in fit
            if False uses the training fit

    '''
    if cross_validation:
        data = copy.copy(fit['psydata']['y']-1)
        model = copy.copy(fit['cv_pred'])
    else:
        data = copy.copy(fit['psydata']['y']-1)
        model = copy.copy(fit['ypred'])

    if plot_this:
        plt.figure()
        alarms,hits,thresholds = metrics.roc_curve(data,model)
        plt.plot(alarms,hits,'ko-')
        plt.plot([0,1],[0,1],'k--')
        plt.ylabel('Hits')
        plt.xlabel('False Alarms')
    return metrics.roc_auc_score(data,model)

