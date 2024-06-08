#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def calc_dist(i,j,points_list):
    d = np.linalg.norm(points_list[i]-points_list[j])
    return d
def d_dstr(pairs,config_l):
    res = []
    for i,j in tqdm(pairs):
        d = []
        for points_list in config_l:
            d.append(calc_dist(i,j,points_list))
        res.append(d)
    return res



def plot_distributions(distributions, labels):

    assert len(np.array(distributions).shape) == len(labels)+1
    assert len(distributions) == len(labels[0])
    colors = ["red", "blue", "green", "orange", "purple", "grey", "yellow"]
    line_type = ["solid", "dotted", "dashed", "dashdot"]
    
    # Create the plot
    fig, ax = plt.subplots()
    
    if len(labels) == 1:
        for i, data in enumerate(distributions):
            hist, bin_edges = np.histogram(data, bins=15, density=True)
            # Plot the histogram as a step plot (outline) by connecting the bin edges
            ax.step(bin_edges[:-1], hist, where='post', lw=2, color=colors[i])
            # Draw last edge
            ax.step([bin_edges[-2], bin_edges[-1]], [hist[-1], hist[-1]],
                    where='post', lw=2, color=colors[i])
        color_legends = [plt.Line2D([0], [0], color=c, linestyle='-', label=cl) 
                         for c, cl in zip(colors[:i+1], labels[0])]
        legend2 = ax.legend(handles=color_legends, title="Colors")
        leg = [legend2]
    
    else:
        assert len(distributions[0]) == len(labels[1])
        for i, group in enumerate(distributions):
            for j, data in enumerate(group):
                hist, bin_edges = np.histogram(data, bins=15, density=True)
                
                ax.step(bin_edges[:-1], hist, where='post', lw=2, linestyle=line_type[j], color=colors[i])
                # Draw last edge
                ax.step([bin_edges[-2], bin_edges[-1]], [hist[-1], hist[-1]],
                        where='post', lw=2, linestyle=line_type[j], color=colors[i])
        
        line_legends = [plt.Line2D([0], [0], color='black', linestyle=lt, label=ll) 
                    for lt, ll in zip(line_type[:j+1], labels[1])]
        color_legends = [plt.Line2D([0], [0], color=c, linestyle='-', label=cl) 
                         for c, cl in zip(colors[:i+1], labels[0])]
        
        legend1 = ax.legend(handles=line_legends, title="Line Types", loc="right")
        legend2 = ax.legend(handles=color_legends, title="Colors")
        leg = [legend1, legend2]
        ax.add_artist(legend1) 
    ax.set_xlabel('Displacement R')
    ax.set_ylabel('Density')
    ax.set_title('Histogram Outlines of Distributions')

    # Return the figure and axes objects to allow for further customization
    return fig, ax, leg

def isotropic_vector_dist(data):
    n = 5
    hist, bin_edges = np.histogram(data, bins=15*n, density=False)
    bin_edges2 = np.diff(bin_edges**3)
    hist2 = hist/bin_edges2
    avgResult = np.average(hist2.reshape(-1, n), axis=1) 
    norm = np.sum(avgResult)
    gap = np.diff(bin_edges[::n])[0]
    return avgResult/norm/gap, bin_edges[::n]

def plot_distributions_vector(distributions, labels):

    assert len(np.array(distributions).shape) == len(labels)+1
    assert len(distributions) == len(labels[0])
    colors = ["red", "blue", "green", "orange", "purple", "grey", "yellow"]
    line_type = ["solid", "dotted", "dashed", "dashdot"]
    
    # Create the plot
    fig, ax = plt.subplots()
    
    if len(labels) == 1:
        for i, data in enumerate(distributions):
            hist, bin_edges = isotropic_vector_dist(data)
            # Plot the histogram as a step plot (outline) by connecting the bin edges
            ax.step(bin_edges[:-1], hist, where='post', lw=2, color=colors[i])
            # Draw last edge
            ax.step([bin_edges[-2], bin_edges[-1]], [hist[-1], hist[-1]],
                    where='post', lw=2, color=colors[i])
        color_legends = [plt.Line2D([0], [0], color=c, linestyle='-', label=cl) 
                         for c, cl in zip(colors[:i+1], labels[0])]
        legend2 = ax.legend(handles=color_legends, title="Colors")
        leg = [legend2]
    
    else:
        assert len(distributions[0]) == len(labels[1])
        for i, group in enumerate(distributions):
            for j, data in enumerate(group):
                hist, bin_edges = isotropic_vector_dist(data)
                
                ax.step(bin_edges[:-1], hist, where='post', lw=2, linestyle=line_type[j], color=colors[i])
                # Draw last edge
                ax.step([bin_edges[-2], bin_edges[-1]], [hist[-1], hist[-1]],
                        where='post', lw=2, linestyle=line_type[j], color=colors[i])
        
        line_legends = [plt.Line2D([0], [0], color='black', linestyle=lt, label=ll) 
                    for lt, ll in zip(line_type[:j+1], labels[1])]
        color_legends = [plt.Line2D([0], [0], color=c, linestyle='-', label=cl) 
                         for c, cl in zip(colors[:i+1], labels[0])]
        
        legend1 = ax.legend(handles=line_legends, title="Line Types", loc="right")
        legend2 = ax.legend(handles=color_legends, title="Colors")
        leg = [legend1, legend2]
        ax.add_artist(legend1) 
    ax.set_xlabel('Displacement R')
    ax.set_ylabel('Density')
    ax.set_title('Histogram Outlines of Distributions')

    # Return the figure and axes objects to allow for further customization
    return fig, ax, leg