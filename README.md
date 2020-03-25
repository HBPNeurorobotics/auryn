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

```bash
sudo apt-get install cmake git build-essential libboost-all-dev
git clone https://ids-git.fzi.de/friedric/erbp.git
cd erbp
pip install -r requirements.txt
pip install -e .
cd build/release
./bootstrap.sh
make -j8
```

Run eRBP on event-based dataset
--------

If you use this code in your paper, please add the following citation:
```
@article{kaiser2020embodied,
  title={Embodied Neuromorphic Vision with Event-Driven Random Backpropagation},
  author={Kaiser, Jacques and Friedrich, Alexander and Tieck, J and Reichard, Daniel and Roennau, Arne and Neftci, Emre and Dillmann, R{\"u}diger},
  year={2020}
}
```

## Download Data Sets:

- IBM DVS Gesture: http://www.research.ibm.com/dvsgesture/
    - copy into folder build/release/experiments/data/dvs_gesture
    - run build/release/experiments/utils/gesture_ds_converter.py
```bash
cd build/release/experiments/

# run with covert attention model (re-indexing events with respect to median in time window):
python2 run_classification_gesture_dual_vis_attention.py

# run with resizing the whole frame:
python2 run_classification_gesture_dual_vis.py
```

## Deprecated Data Sets:

- MNIST-DVS: http://www2.imse-cnm.csic.es/caviar/MNISTDVS.html
    - copy into folder build/release/experiments/data/dvs_mnist_saccade
```bash
cd build/release/experiments/
python2 run_classification_mnist_saccade.py
```
- MNIST-FLASH-DVS: http://www2.imse-cnm.csic.es/caviar/MNISTDVS.html
    - copy into folder build/release/experiments/data/dvs_mnist_flash
```bash
cd build/release/experiments/
python2 run_classification_mnist_flash.py
```


Most important files
--------
- **Read aedat files**: build/release/experiments/utils/jaer_data_handler.py
- **Generate ras**: build/release/experiments/utils/generate_ras.py
- **Run experiment**: build/release/experiments/run_classification_*.py
- **Evaluate experiment**: build/release/experiments/experimentLib.py
- **Plotting**: build/release/experiments/utils/erbp_plotter.py
- **Experiment network**: experiments/exp_rbp_flash.cpp

License & Copyright (eRBP DVS)
---------------------------

Copyright 2017-2020 FZI

eRBP for DVS processing scripts under Auryn is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

eRBP DVS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Auryn.  If not, see <http://www.gnu.org/licenses/>.


License & Copyright (eRBP)
---------------------------

Copyright 2016-2017 Emre Neftci

eRBP scripts under Auryn  is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

eRBP is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Auryn.  If not, see <http://www.gnu.org/licenses/>.


License & Copyright (Auryn)
---------------------------

Copyright 2014-2017 Friedemann Zenke

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
