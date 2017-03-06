/* 
* Copyright 2014-2016 Friedemann Zenke
*
* This file is part of Auryn, a simulation package for plastic
* spiking neural networks.
* 
* Auryn is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* Auryn is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
*/

/*!\file 
 *
 * \brief Simulation code for the Vogels Abbott benchmark following Brette et al. (2007)
 *
 * This simulation implements the Vogels Abbott benchmark as suggested by
 * Brette et al. (2007) Journal of Computational Neuroscience 23: 349-398. 
 *
 * The network is based on a network by Vogels and Abbott as described in 
 * Vogels, T.P., and Abbott, L.F. (2005).  Signal propagation and logic gating
 * in networks of integrate-and-fire neurons. J Neurosci 25, 10786.
 *
 * We used this network for benchmarking Auryn against other simulators in
 * Zenke, F., and Gerstner, W. (2014). Limits to high-speed simulations of
 * spiking neural networks using general-purpose computers. Front Neuroinform
 * 8, 76.
 *
 * See build/release/run_benchmark.sh for automatically run benchmarks to
 * compare the performance of different Auryn builds.
 *
 * */

#include "auryn.h"
#include <stdio.h>
#include <stdlib.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>

using namespace auryn;

namespace po = boost::program_options;

int main(int ac,char *av[]) {
	string dir = "./outputs/test_srm/";

	std::stringstream oss;
	string strbuf ;
	string msg;

	double simtime = 100.;
  double sigma = 50e-3;

	NeuronID nc1 = 392/2;
	NeuronID nc2 = 392/4;
	NeuronID nvis = 794;

	int errcode = 0;
  
	auryn_init( ac, av, dir );
	sys->set_simulation_name("test_erbp_conv2d");

	oss << dir  << "/erbp_conv2d." << sys->mpi_rank() << ".";
	string outputfile = oss.str();

	logger->msg("Setting up neuron groups ...",PROGRESS,true);

  string ip_v = "/tmp/input_v";
  string fc1 = "/tmp/layer1_cw.mtx";
  string fc2 = "/tmp/layer2_cw.mtx";

  int nfeat1 = 4;
  int nfeat2 = 8;
  
	SIFGroup * neurons_c1[nfeat1];
  for(int i=0; i<nfeat1; i++){
    neurons_c1[i]=new SIFGroup(nc1);
  }

	SIFGroup * neurons_c2[nfeat2];
  for(int i=0; i<nfeat2; i++){
    neurons_c2[i]=new SIFGroup(nc2);
  }
  //tau_mem^-1 scale because patternstimulator does not multiply with dt/tau_mem but dt.
  FileInputGroup * neurons_vis = new FileInputGroup(nvis, ip_v, true);
  //  }

  std::stringstream filename;


  peRBPSharedConnection * conv1[nfeat1];
  for(int i=0; i<nfeat1; i++){
	filename.str("");
	filename.clear();
	filename << "/tmp/layer1_w_" << i << ".mtx"; 
  conv1[i] = new peRBPSharedConnection(
      neurons_vis,
      neurons_c1[i],
      fc1.c_str(),
      filename.str().c_str(),
      1.0,
      MEM);
  }
  
  peRBPSharedConnection * conv2[nfeat2*nfeat1];
  for(int i=0; i<nfeat1; i++){
  for(int j=0; j<nfeat2; j++){
	filename.str("");
	filename.clear();
	filename << "/tmp/layer2_w_" << i << "_" << j << ".mtx"; 
  conv2[i*nfeat2+j] = new peRBPSharedConnection(
      neurons_c1[i],
      neurons_c2[j],
      fc2.c_str(),
      filename.str().c_str(),
      1.0,
      MEM);
  }
  }

//	neurons_hid->set_refractory_period(4.0e-3); // minimal ISI 5.1ms

//	PoissonStimulator ns = PoissonStimulator( neurons_hid, 1000., sigma);

	msg = "Setting up monitors ...";
	logger->msg(msg,PROGRESS,true);

	filename.str("");
	filename.clear();
	filename << outputfile << "vis.ras";
	SpikeMonitor * smon_vis = new SpikeMonitor( neurons_vis, filename.str().c_str() );


	SpikeMonitor * smon_c1[nfeat1];
  for(int i=0; i<nfeat1; i++){
	filename.str("");
	filename.clear();
	filename << outputfile << "c1" << i << ".ras";
  smon_c1[i] = new SpikeMonitor( neurons_c1[i], filename.str().c_str() );
  }

	SpikeMonitor * smon_c2[nfeat2];
  for(int i=0; i<nfeat2; i++){
	filename.str("");
	filename.clear();
	filename << outputfile << "c2" << i << ".ras";
  smon_c2[i] = new SpikeMonitor( neurons_c2[i], filename.str().c_str() );
  }


	BinaryStateMonitor * mem_mon = new BinaryStateMonitor( neurons_c1[0], 1, "mem", "yes" );

	logger->msg("Simulating ..." ,PROGRESS,true);
	if (!sys->run(simtime,true)) 
			errcode = 1;

	logger->msg("Freeing ..." ,PROGRESS,true);
	//auryn_free();
  logger->msg("Saving network state ...",PROGRESS,true);
  sys->save_network_state_text(outputfile);
	if (errcode)
		auryn_abort(errcode);
	return errcode;
}
