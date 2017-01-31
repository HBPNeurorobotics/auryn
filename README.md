![Auryn logo](http://www.fzenke.net/uploads/images/logo_trans_small.png "Auryn logo")

Auryn 
=====

Auryn is Simulator for recurrent spiking neural networks with synaptic
plasticity. It comes with the GPLv3 (please see COPYING).

* For examples and documentation visit http://www.fzenke.net/auryn/
* Please reporte issues here https://github.com/fzenke/auryn/issues
* For questions and support http://www.fzenke.net/auryn/forum/

This is a fork of Auryn implementing event-driven random backpropagation for spike-based deep learning.

Quick start
-----------

Note, Auryn needs a C++ compiler, the boost libraries (www.boost.org) with MPI
support installed. To download and compile the examples under Ubuntu Linux try:

```
sudo apt-get install cmake git build-essential libboost-all-dev
git clone git@gitlab.com:NMI-lab/erbp_auryn.git && cd auryn/build/release
./bootstrap.sh && make
```

Run eRBP
--------

```
cd build/release/experiments/
python2 run_classification_mnist_deep_dual.py
```

This will train a two-layer feed-forward spiking neural network (all to all connections) on MNIST digits as described in <https://arxiv.org/abs/1612.05596>.



License & Copyright 
-------------------

Copyright 2014-2016 Friedemann Zenke,
Copyright 2017 Emre Neftci (on eRBP specific work)

Auryn is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Auryn is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
