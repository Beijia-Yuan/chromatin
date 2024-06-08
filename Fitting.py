#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from tqdm import tqdm
import mpmath
import random

def get_norm(pdf):
    intervals = [(i*10,i*10+10) for i in range(10)]
    norm = 0
    for a, b in intervals:
        new, error = quad(pdf, a, b)
        if new != 0:
            norm += new
        elif norm != 0:
            break
    return float(norm+1e-40)

def mean(f):
    """Compute the mean of the distribution."""
    result, error = quad(lambda x: x * f(x), 0, np.inf)
    return result

def var(f):
    """Compute the variance of the distribution."""
    mu = mean(f)
    result, error = quad(lambda x: (x - mu)**2 * f(x), 0, np.inf)
    return result

def mean_variance(f,a,b):
    def P(x):
        return f(x,a,b)
    norm = get_norm(P)
    
    def pdf(x):
        return float(P(x)/norm)
    return mean(pdf), var(pdf)

def error_function(f):
    # error function returns the squared distance between predicted and actual mean and variance
    def ef(params, m, var):
        a, b = params
        c_pred, d_pred = mean_variance(f,a,b)
        error = (c_pred - m)**2 + (d_pred - var)**2
        return error
    return ef

def fit_mean_var(dl, f_generator, *args, **kwargs): 
    # dl stores the end-to-end distance for samples, dim = (different v's, different pairs, distance list)
    fitted_ab = [[(0,0)]*len(dl[0])] # assume that the first entry is v = 0 thus no need to be fitted
    for i in tqdm(range(1,len(dl))):
        fitted_ab.append([])
        for j in (range(len(dl[i]))):
            f = f_generator(i, j, *args, **kwargs)
            objective_function = error_function(f)
            m = np.mean(dl[i][j]); var = np.var(dl[i][j])
            initial_guess = [3, 5]
            
            while True:
                result = minimize(objective_function, initial_guess, 
                                  args=(m, var), bounds = ((0, None), (0, None)))
                if result.fun < 1e-5:
                    break
                else:
                    initial_guess[0] = initial_guess[0]* random.randint(2,10)
                    initial_guess[1] = initial_guess[1]*random.randint(2,10)
                    #print(initial_guess)
                
            
            a_best, b_best = result.x
            fitted_ab[i].append((a_best, b_best))
    return fitted_ab

def generate_gaussian(i, j):
    def gaussian(x,a,b):
        return x**2*np.exp(-(x-a)**2/(2*b**2+1e-40))
    return gaussian

def generate_model(i, j, pairs):
    Neff = pairs[j][1]- pairs[j][0]
    def model(x,a,b):
        E = 3/(2*Neff)*x**2 + a*100**2*(x**2+b)**(-3/2)
        return x**2*np.exp(-E)
    return model

def generate_gaussian_vector(i, j):
    def gaussian(x,a,b):
        return np.exp(-(x-a)**2/(2*b**2+1e-40))
    return gaussian

def generate_model_vector(i, j, pairs):
    Neff = pairs[j][1]- pairs[j][0]
    def model(x,a,b):
        E = 3/(2*Neff)*x**2 + a*100**2*(x**2+b)**(-3/2)
        return np.exp(-E)
    return model

def pdf_generator(f,a,b):
    def P(x):
        return f(x,a,b)
    norm = get_norm(P)
    
    def pdf(x):
        return (P(x)/norm)
    return pdf


def plot_prediction(ax, f_generator, params, ls, *args, **kwargs):
    colors = ["red","blue","green", "orange", "purple", "grey", "yellow"]
    x = np.linspace(1,ax.get_xlim()[1],200)
    for i, (a, b) in enumerate(params):
        f = f_generator(i,*args, **kwargs)
        pdf = pdf_generator(f,a,b)
        ax.plot(x, pdf(x), color = colors[i], linestyle=ls)