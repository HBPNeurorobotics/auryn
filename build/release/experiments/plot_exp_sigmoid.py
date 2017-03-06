#!/bin/python
#-----------------------------------------------------------------------------
# File Name : plot_exp_sigmoid.py
# Author: Emre Neftci
#
# Creation Date : Tue 03 Jan 2017 10:31:03 AM PST
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from pylab import *
from experimentLib import *

SL = monitor_to_spikelist('outputs/exp_sigmoid/exp_sigmoid.0.hid.ras')
SL.t_stop = 50000
SL.t_start = 0
SL.complete(range(100))

inp = np.linspace(-50,49,100)*6e-3
plot(inp,SL.mean_rates())
show()
