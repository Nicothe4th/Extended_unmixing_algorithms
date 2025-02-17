#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:04:45 2025

@author: jnmc
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
errors_mmm_python = np.load('errors_mmm_python.npy')  # Cargar
data_matlab = loadmat('errors_mmm_matlab.mat')

E_zm = data_matlab['E_zm']
E_am = data_matlab['E_am']
E_pm = data_matlab['E_pm']
E_sm = data_matlab['E_sm']
E_tm = data_matlab['E_tm']

E_zp = errors_mmm_python[0,:,:]
E_ap = errors_mmm_python[1,:,:]
E_pp = errors_mmm_python[2,:,:]
E_sp = errors_mmm_python[3,:,:]
E_tp = errors_mmm_python[4,:,:]


# Functions list (names of the methods)
functions = ['NEBEAE', 'NEBEAE-TV', 'NEBEAESN', 'NEBEAESNTV', 'NESSEAE']

# Titles for each boxplot (one for each error metric)
titles = ['Reconstruction Error', 'Abundance Error', 'Endmember Error', 'SAM Error', 'Computation Time']

plt.figure(1, figsize=(17, 10))
plt.clf()

# Loop to create subplots for each method (rows)
for i, function in enumerate(functions):
    for j, (E_m, E_p, title) in enumerate(zip([E_zm, E_am, E_pm, E_sm, E_tm], 
                                               [E_zp, E_ap, E_pp, E_sp, E_tp], titles)):
        ax = plt.subplot(5, 5, i * 5 + j + 1)  # Grid layout: 5 rows, 5 columns
        
        # Combine MATLAB and Python data for the current error metric and method
        data = [E_m[i, :], E_p[i, :]]  # MATLAB and Python data for the current method and error metric
        
        # Boxplot for MATLAB (first) and Python (second) data
        ax.boxplot(data, notch=True, showfliers=False)
        
        # Set x-tick labels for 'Matlab' and 'Python'
        ax.set_xticklabels(['Matlab', 'Python'])
        
        # Add text label on the left of the first column (method names)
        if j == 0:  # Only add method label on the first column
            ax.annotate(function, xy=(-0.3, 0.5), xycoords='axes fraction', horizontalalignment='center',
                        verticalalignment='center', rotation=90, fontsize=12, weight='bold')
        
        # Set title for the boxplot
        if i == 0:  # Only add the error metric title on the first row
            ax.set_title(title)

# Adjust layout
plt.tight_layout()
plt.show()
