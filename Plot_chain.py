#!/usr/bin/env python
# coding: utf-8


from matplotlib import pyplot as plt
import numpy as np

def plot_chain(chain):
    # Generating 100 random 3D points (replace this with your actual data)
    np.random.seed(0)  # For reproducibility
    points = chain.state

    # Extracting x, y, z coordinates for plotting
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Creating a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting points
    ax.scatter(x, y, z)

    # Connecting points with lines to form a chain
    ax.plot(x, y, z, color='red')
    
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


    # Setting labels (optional)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Show plot
    plt.show()

