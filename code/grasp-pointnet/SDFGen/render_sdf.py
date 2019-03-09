import math
import numpy as np
from scipy import linalg
from scipy import spatial
import sys

import logging
import matplotlib as mp
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
import pylab as plt

import IPython

DEF_SDF_FILE = '../Teapot N260113.sdf'
SMALL_SDF_FILE = '../Teapot small N290514.sdf'

class SDF(object):
    def __init__(self, dims, origin, dx, sdf_values):
        self.dims = dims
        self.origin = origin
        self.dx = dx
        self.sdf_values = sdf_values

def read_sdf(filename):
    '''
    Returns a 3d numpy array of SDF values from the input file
    '''
    try:
        f = open(filename, 'r')
    except IOError:
        logging.error('Failed to open sdf file: %s' %(filename))
        return None
    dim_str = f.readline()
    origin_str = f.readline()
    dx_str = f.readline()
    
    # convert header info to floats
    i = 0
    dims = np.zeros([3])
    for d in dim_str.split(' '):
        dims[i] = d
        i = i+1

    i = 0
    origin = np.zeros([3])
    for x in origin_str.split(' '):
        origin[i] = x
        i = i+1

    dx = float(dx_str)

    # read in all sdf values
    sdf_grid = np.zeros(dims)
    i = 0
    j = 0
    k = 0
    for line in f:
        if k < dims[2]:
            sdf_grid[i,j,k] = float(line)
            i = i + 1
            if i == dims[0]:
                i = 0
                j = j+1
            if j == dims[1]:
                j = 0
                k = k+1
    sdf = SDF(dims, origin, dx, sdf_grid)
    return sdf

def render_sdf(sdf, thresh = 0.01):
    h = plt.figure()
    ax = h.add_subplot(111, projection = '3d')

    surface_points = np.where(np.abs(sdf.sdf_values) < thresh)

    x = surface_points[0]
    y = surface_points[1]
    z = surface_points[2]


    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))
    
    X, Y = np.meshgrid(xi, yi)
    Z = griddata(x, y, z, xi, yi)

    ax.scatter(surface_points[0], surface_points[1], surface_points[2])
#    ax.plot_surface(X, Y, Z, rstride = 6, cstride = 6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(0,sdf.dims[0])
    ax.set_ylim3d(0,sdf.dims[1])
    ax.set_zlim3d(0,sdf.dims[2])

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        sdf_filename = sys.argv[1]
    else:
        sdf_filename = DEF_SDF_FILE
    sdf = read_sdf(sdf_filename)
    render_sdf(sdf, 0.05)
