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

using namespace auryn;

namespace po = boost::program_options;

int main(int ac,char *av[]) {
  bool record_rates;
  bool record;
  bool record_rasters;
  bool learning;

	string dir = "/tmp";

	string fwmat_ve = "";
	string fwmat_vh = "";
	string fwmat_c1 = "";
	string fwmat_c2 = "";
	string fwmat_ck1 = "";
	string fwmat_ck2 = "";
	string fwmat_ec1 = "";
	string fwmat_ec2 = "";
	string fwmat_ho = "";
	string fwmat_hh = "";
  string fwmat_oe = "";
  string fwmat_eo = "";
  string fwmat_eh = "";
  string ip_v = "";

	std::stringstream oss;
	string strbuf ;
	string msg;

	double wmax = 32768;
	double simtime = 20.;
	double stimtime = 10.;
	double eta = 0.;
  double sigma = 50e-3;
  double prob_syn = 1.0;
  double gate_lo = -15.;
  double gate_hi = 15.;

	NeuronID nout = 10;
	NeuronID nvis = 794;
	NeuronID nc1 = (nvis-nout)/4;
	NeuronID nc2 = nc1/4;
	NeuronID nhid = 100;
  NeuronID nfeat1 = 32;
  NeuronID nfeat2 = 64;


  std::stringstream filename;
	int errcode = 0;


    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("simtime", po::value<double>(), "duration of simulation")
            ("dir", po::value<string>(), "dir from file")
            ("fve", po::value<string>(), "file with visible to error connections")
            ("fvh", po::value<string>(), "file with last conv to hidden connections")
            ("fc1", po::value<string>(), "adjacency conv layer 1")
            ("fc2", po::value<string>(), "kernel conv layer 1")
            ("fck1", po::value<string>(), "prefix adjacency conv layer 2")
            ("fck2", po::value<string>(), "prefix kernel conv layer 2")
            ("fec1", po::value<string>(), "prefix for error connections conv1")
            ("fec2", po::value<string>(), "prefix for error connections conv2")
            ("fho", po::value<string>(), "file with hidden to visible connections")
            ("fhh", po::value<string>(), "file with hidden to hidden connections")
            ("foe", po::value<string>(), "file with output to error connections")
            ("feo", po::value<string>(), "file with error to output connections")
            ("feh", po::value<string>(), "file with error to hidden connections")
            ("ip_v", po::value<string>(), "file with input patterns for visible layer")
            ("sigma", po::value<double>(), "poisson stimulator weight")
            ("stimtime", po::value<double>(), "stimtime")
            ("eta", po::value<double>(), "learning rate")
            ("learn", po::value<bool>(), "learning active (faster when false)")
            ("record_full", po::value<bool>(), "monitors active (faster when false)")
            ("record_rates", po::value<bool>(), "record rates")
            ("record_rasters", po::value<bool>(), "record rasters")
            ("nvis", po::value<int>(), "Number of visible units")
            ("nout", po::value<int>(), "Number of output units")
            ("nhid", po::value<int>(), "Number of hidden units")
            ("nfeat1", po::value<int>(), "Number first layer features")
            ("nfeat2", po::value<int>(), "Number second layer features")
            ("prob_syn", po::value<double>(), "tranmission probability of plastic synapses")
            ("gate_low", po::value<double>(), "Gating value (low) of plastic synapses")
            ("gate_high", po::value<double>(), "Gating value (high) of plastic synapses")

        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        if (vm.count("simtime")) simtime = vm["simtime"].as<double>(); 
        if (vm.count("stimtime")) stimtime = vm["stimtime"].as<double>(); 
        if (vm.count("dir")) dir = vm["dir"].as<string>(); 
        if (vm.count("fve")) fwmat_ve = vm["fve"].as<string>();
        if (vm.count("fvh")) fwmat_vh = vm["fvh"].as<string>();
        if (vm.count("fc1")) fwmat_c1 = vm["fc1"].as<string>();
        if (vm.count("fc2")) fwmat_c2 = vm["fc2"].as<string>();
        if (vm.count("fck1")) fwmat_ck1 = vm["fck1"].as<string>();
        if (vm.count("fck2")) fwmat_ck2 = vm["fck2"].as<string>();
        if (vm.count("fec1")) fwmat_ec1 = vm["fec1"].as<string>();
        if (vm.count("fec2")) fwmat_ec2 = vm["fec2"].as<string>();
        if (vm.count("fho")) fwmat_ho = vm["fho"].as<string>();
        if (vm.count("fhh")) fwmat_hh = vm["fhh"].as<string>();
        if (vm.count("foe")) fwmat_oe = vm["foe"].as<string>();
        if (vm.count("feo")) fwmat_eo = vm["feo"].as<string>();
        if (vm.count("feh")) fwmat_eh = vm["feh"].as<string>();
        if (vm.count("learn")) learning = vm["learn"].as<bool>();
        if (vm.count("record_full")) record = vm["record_full"].as<bool>();
        if (vm.count("record_rates")) record_rates = vm["record_rates"].as<bool>();
        if (vm.count("record_rasters")) record_rasters = vm["record_rasters"].as<bool>();
        if (vm.count("eta")) eta = vm["eta"].as<double>();
        if (vm.count("sigma")) sigma = vm["sigma"].as<double>();
        if (vm.count("prob_syn")) prob_syn = vm["prob_syn"].as<double>();
        if (vm.count("gate_low")) gate_lo = vm["gate_low"].as<double>();
        if (vm.count("gate_high")) gate_hi = vm["gate_high"].as<double>();
        if (vm.count("nvis")) nvis = vm["nvis"].as<int>();
        if (vm.count("nhid")) nhid = vm["nhid"].as<int>();
        if (vm.count("nfeat1")) nfeat1 = vm["nfeat1"].as<int>();
        if (vm.count("nfeat1")) nfeat2 = vm["nfeat2"].as<int>();
        if (vm.count("nout")) nout = vm["nout"].as<int>();
  	    if (vm.count("ip_v")) ip_v  = vm["ip_v"].as<string>();

    }
    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }


	auryn_init( ac, av, dir );
	sys->set_simulation_name("exp_rbp");

	oss << dir  << "/coba." << sys->mpi_rank() << ".";
	string outputfile = oss.str();

	logger->msg("Setting up neuron groups ...",PROGRESS,true);


	SIFGroup * neurons_out = new SIFGroup( nout);
	LinGroup * neurons_err1 = new LinGroup( nout);
	LinGroup * neurons_err2 = new LinGroup( nout);
	SRM0Group * neurons_vis = new SRM0Group( nvis);
  PatternStimulator * stim = new PatternStimulator(neurons_vis, ip_v, "ipat", 1., nvis);
	SIFGroup * neurons_c1[nfeat1]; 
	SIFGroup * neurons_c2[nfeat2]; 
  PoissonStimulator * ps_c1[nfeat1];
  PoissonStimulator * ps_c2[nfeat2];
  for(int i=0; i<nfeat1; i++){ 
    neurons_c1[i] = new SIFGroup(nc1, RANKLOCK);
    neurons_c1[i]->set_refractory_period(3.9e-3);
    neurons_c1[i]->set_bg_currents(0.0e-3);
    neurons_c1[i]->set_bg_currents_dendrite(0e-2);
    if (sigma>0) ps_c1[i] = new PoissonStimulator( neurons_c1[i], 1000., sigma);
  }
  for(int i=0; i<nfeat2; i++){ 
    neurons_c2[i] = new SIFGroup(nc2, RANKLOCK);
    neurons_c2[i]->set_refractory_period(3.9e-3);
    neurons_c2[i]->set_bg_currents(0.0e-3);
    neurons_c2[i]->set_bg_currents_dendrite(0e-2);
    if (sigma>0) ps_c2[i] = new PoissonStimulator( neurons_c2[i], 1000., sigma);
  }
	SIFGroup * neurons_hid = new SIFGroup( nhid);
	neurons_vis->set_refractory_period(4.0e-3);


  if(sigma>0){
	PoissonStimulator * ps_hid = new PoissonStimulator( neurons_hid, 1000., sigma);
	PoissonStimulator * ps_out = new PoissonStimulator( neurons_out, 1000., sigma);
  } else {
	logger->msg("No PoissonStimulator ...",PROGRESS,true);
  }
	printf("prob syn %f", prob_syn);

	neurons_out->set_refractory_period(3.9e-3);
	neurons_hid->set_refractory_period(3.9e-3); // minimal ISI 5.1ms

	neurons_out->set_bg_currents(0.0e-3); // corresponding to 200pF for C=200pF and tau=20ms
	neurons_out->set_bg_currents_dendrite(0e-2);
	neurons_hid->set_bg_currents(0.0e-3); // corresponding to 200pF for C=200pF and tau=20ms
	neurons_hid->set_bg_currents_dendrite(0e-2);
	neurons_err1->set_bg_currents(0.0e-3); // corresponding to 200pF for C=200pF and tau=20ms
	neurons_err2->set_bg_currents(0.0e-3); // corresponding to 200pF for C=200pF and tau=20ms

	logger->msg("Setting up E connections ...",PROGRESS,true);

	peRBPConnection * con_ho 
		= new peRBPConnection(
            neurons_hid,
            neurons_out,
            fwmat_ho.c_str(),
            prob_syn,
            eta, //eta
            gate_lo,
            gate_hi,
            AMPA);

  peRBPSharedConnection * conv1[nfeat1];
  for(int i=0; i<nfeat1; i++){
	filename.str("");
	filename.clear();
	filename << fwmat_ck1 << "_" << i << ".mtx"; 
  conv1[i] = new peRBPSharedConnection(
      neurons_vis,
      neurons_c1[i],
      fwmat_c1.c_str(),
      filename.str().c_str(),
      prob_syn,
      eta,
      gate_lo,
      gate_hi,
      AMPA);
  }
  
  peRBPSharedConnection * conv2[nfeat2*nfeat1];
  for(int i=0; i<nfeat1; i++){
  for(int j=0; j<nfeat2; j++){
	filename.str("");
	filename.clear();
	filename << fwmat_ck2 << "_" << i << "_" << j << ".mtx"; 
  conv2[i*nfeat2+j] = new peRBPSharedConnection(
      neurons_c1[i],
      neurons_c2[j],
      fwmat_c2.c_str(),
      filename.str().c_str(),
      prob_syn,
      eta,
      gate_lo,
      gate_hi,
      AMPA);
  }
  }
  
  peRBPConnection * con_vh[nfeat2];
  for(int i=0; i<nfeat2; i++){
	filename.str("");
	filename.clear();
	filename << fwmat_vh << "_" << i << ".mtx"; 
	con_vh[i]
		= new peRBPConnection(
            neurons_c2[i],
            neurons_hid,
            filename.str().c_str(),
            prob_syn,
            eta, //eta
            gate_lo,
            gate_hi,
            AMPA);
  }



if (eta>0){
	SparseConnection * con_ve1
		= new SparseConnection( neurons_vis, neurons_err1,
            fwmat_ve.c_str(),
            MEM);
  con_ve1->set_gain(1);

	SparseConnection * con_ve2
		= new SparseConnection( neurons_vis, neurons_err2,
            fwmat_ve.c_str(),
            MEM);
  con_ve2->set_gain(-1);

	SparseConnection * con_oe1 
		= new SparseConnection( neurons_out, neurons_err1,
            fwmat_oe.c_str(),
            MEM);
  con_oe1->set_gain(1);

	SparseConnection * con_oe2 
		= new SparseConnection( neurons_out, neurons_err2,
            fwmat_oe.c_str(),
            MEM);
  con_oe2->set_gain(-1);

	SparseConnection * con_e1o 
		= new SparseConnection( neurons_err1, neurons_out,
            fwmat_eo.c_str(),
            DEND);
  con_e1o->set_gain(1);

	SparseConnection * con_e2o 
		= new SparseConnection( neurons_err2, neurons_out,
            fwmat_eo.c_str(),
            DEND);
  con_e2o->set_gain(-1);

	SparseConnection * con_e1h 
		= new SparseConnection( neurons_err1, neurons_hid,
            fwmat_eh.c_str(),
            DEND);
  con_e1h->set_gain(1);

	SparseConnection * con_e2h 
		= new SparseConnection( neurons_err2, neurons_hid,
            fwmat_eh.c_str(),
            DEND);
  con_e2h->set_gain(-1);

	SparseConnection * con_e1c1[nfeat1]; 
	SparseConnection * con_e2c1[nfeat1]; 
  for(int i=0; i<nfeat1; i++){
    filename.str("");
    filename.clear();
    filename << fwmat_ec1 << "_" << i << ".mtx";
    con_e1c1[i] = new SparseConnection( neurons_err1, neurons_c1[i],
              filename.str().c_str(),
              DEND);
    con_e1c1[i]->set_gain(1);
    con_e2c1[i] = new SparseConnection( neurons_err2, neurons_c1[i],
              filename.str().c_str(),
              DEND);
    con_e2c1[i]->set_gain(-1);
  }

	SparseConnection * con_e1c2[nfeat2]; 
	SparseConnection * con_e2c2[nfeat2]; 
  for(int i=0; i<nfeat2; i++){
    filename.str("");
    filename.clear();
    filename << fwmat_ec2 << "_" << i << ".mtx";
    con_e1c2[i]	= new SparseConnection( neurons_err1, neurons_c2[i],
              filename.str().c_str(),
              DEND);
    con_e1c2[i]->set_gain(1);
    con_e2c2[i]	= new SparseConnection( neurons_err2, neurons_c2[i],
              filename.str().c_str(),
              DEND);
    con_e2c2[i]->set_gain(-1);
  }
  }

  if (!fwmat_hh.empty()){
	peRBPConnection * con_hh 
		= new peRBPConnection(
            neurons_hid,
            neurons_hid,
            fwmat_hh.c_str(),
            prob_syn,
            eta, //eta
            gate_lo,
            gate_hi,
            AMPA);
  }

  if (record){
	BinaryStateMonitor * dend_mon = new BinaryStateMonitor( neurons_out, 0, "dendrite", sys->fn("bdendrite") );
	BinaryStateMonitor * mem_mon = new BinaryStateMonitor( neurons_out, 0, "mem", sys->fn("bmem") );
	BinaryStateMonitor * vis_mem_mon = new BinaryStateMonitor( neurons_vis, 127, "bg_current", sys->fn("bmemvis") );
	BinaryStateMonitor * g_ampa_mon = new BinaryStateMonitor( neurons_out, 0, "g_ampa", sys->fn("bampa") );
	BinaryStateMonitor * errmon = new BinaryStateMonitor( neurons_err1, 0, "mem", sys->fn("mem") );
  oss.str("");
	oss << outputfile << "v.w";
	WeightMonitor * smon = new WeightMonitor( con_ho, oss.str(), auryn_timestep);
  smon->add_to_list(32,0);
  }

	if ( record_rasters ) {
		msg = "Setting up monitors ...";
		logger->msg(msg,PROGRESS,true);
		filename.str("");
		filename.clear();
		filename << outputfile << "err2.ras";
		SpikeMonitor * smon_err2 = new SpikeMonitor( neurons_err2, filename.str().c_str() );
		filename.str("");
		filename.clear();
		filename << outputfile << "err1.ras";
		SpikeMonitor * smon_err1 = new SpikeMonitor( neurons_err1, filename.str().c_str() );
		filename.str("");
		filename.clear();
		filename << outputfile << "hid.ras";
		SpikeMonitor * smon_hid = new SpikeMonitor( neurons_hid, filename.str().c_str() );

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

		filename.str("");
		filename.clear();
		filename << outputfile << "vis.ras";
		SpikeMonitor * smon_vis
      = new SpikeMonitor( neurons_vis, filename.str().c_str() );
	}

  if ( record_rates ) {
		filename.str("");
		filename.clear();
		filename << outputfile << "out.ras";
		SpikeMonitor * smon_out = new SpikeMonitor( neurons_out, filename.str().c_str() );
  }






//   filename.str("");
//   filename.clear();
//   filename << outputfile << "out.rates";
//   RateMonitor * rmon_out = new RateMonitor( neurons_out, filename.str().c_str(), .1);
//


	logger->msg("Simulating ..." ,PROGRESS,true);
	if (!sys->run(simtime,true)) 
			errcode = 1;

	if ( sys->mpi_rank() == 0 ) {
		logger->msg("Saving elapsed time ..." ,PROGRESS,true);
		char filenamebuf [255];
		sprintf(filenamebuf, "%s/elapsed.dat", dir.c_str());
		std::ofstream timefile;
		timefile.open(filenamebuf);
		timefile << sys->get_last_elapsed_time() << std::endl;
		timefile.close();
	}

	logger->msg("Saving network state ...",PROGRESS,true);
  sys->save_network_state_text(outputfile);
	if (errcode)
		auryn_abort(errcode);

	logger->msg("Freeing ..." ,PROGRESS,true);
	auryn_free();
	return errcode;
}
