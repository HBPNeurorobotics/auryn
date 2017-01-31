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
	NeuronID nhid = 100;


	int errcode = 0;


    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("simtime", po::value<double>(), "duration of simulation")
            ("dir", po::value<string>(), "dir from file")
            ("fve", po::value<string>(), "file with visible to error connections")
            ("fvh", po::value<string>(), "file with visible to hidden connections")
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
	sys->set_simulation_name("exp_rbp_online");

	oss << dir  << "/coba." << sys->mpi_rank() << ".";
	string outputfile = oss.str();

	logger->msg("Setting up neuron groups ...",PROGRESS,true);


	LinGroup * neurons_err = new LinGroup( nout);
	SRM0Group * neurons_vis = new SRM0Group( nvis);
	SIFGroup * neurons_out = new SIFGroup( nout);
	SIFGroup * neurons_hid = new SIFGroup( nhid);

	logger->msg("Done setting up neuron groups ...",PROGRESS,true);
	logger->msg("Setting up Pattern Stimulator ...",PROGRESS,true);
  PatternStimulator * stim = new PatternStimulator(neurons_vis, ip_v, "ipat", 1., nvis);
	logger->msg("Done setting up Pattern Stimulator ...",PROGRESS,true);

  if(sigma>0){
	PoissonStimulator * ps_hid = new PoissonStimulator( neurons_hid, 1000., sigma);
	PoissonStimulator * ps_out = new PoissonStimulator( neurons_out, 1000., sigma);
  } else {
	logger->msg("No PoissonStimulator ...",PROGRESS,true);
  }
	printf("prosyb %f", prob_syn);

	neurons_out->set_refractory_period(4.0e-3);
	neurons_vis->set_refractory_period(4.0e-3);
	neurons_hid->set_refractory_period(4.0e-3); // minimal ISI 5.1ms

	neurons_out->set_bg_currents(0.0e-3); // corresponding to 200pF for C=200pF and tau=20ms
	neurons_out->set_bg_currents_dendrite(-25e-2);
	neurons_hid->set_bg_currents(0.0e-3); // corresponding to 200pF for C=200pF and tau=20ms
	neurons_hid->set_bg_currents_dendrite(0e-2);
	neurons_err->set_bg_currents(500*100.0e-3); // corresponding to 200pF for C=200pF and tau=20ms

	logger->msg("Setting up E connections ...",PROGRESS,true);

	modulatedVmPlasticConnection * con_vh
		= new modulatedVmPlasticConnection( neurons_vis,neurons_hid,
            fwmat_vh.c_str(),
            eta, //eta
		        20e-3,
			      20e-3,
            wmax,
            AMPA,
            prob_syn,
            gate_lo,
            gate_hi);

	modulatedVmPlasticConnection * con_ho 
		= new modulatedVmPlasticConnection( neurons_hid,neurons_out,
            fwmat_ho.c_str(),
            eta, //eta
		        20e-3,
			      20e-3,
            wmax,
            AMPA,
            prob_syn,
            gate_lo,
            gate_hi);


	SparseConnection * con_ve
		= new SparseConnection( neurons_vis,neurons_err,
            fwmat_ve.c_str(),
            MEM);

	Connection * con_oe 
		= new SparseConnection( neurons_out,neurons_err,
            fwmat_oe.c_str(),
            MEM);

  if (eta>0){
	Connection * con_eo 
		= new SparseConnection( neurons_err,neurons_out,
            fwmat_eo.c_str(),
            DEND);

	Connection * con_eh 
		= new SparseConnection( neurons_err,neurons_hid,
            fwmat_eh.c_str(),
            DEND);
  }

  if (!fwmat_hh.empty()){
	modulatedVmPlasticConnection * con_hh 
		= new modulatedVmPlasticConnection( neurons_hid,neurons_hid,
            fwmat_hh.c_str(),
            eta, //eta
		        20e-3,
			      20e-3,
            wmax,
            AMPA,
            prob_syn);
  }

  std::stringstream filename;
  if (record){
	BinaryStateMonitor * dend_mon = new BinaryStateMonitor( neurons_out, 0, "dendrite", sys->fn("bdendrite") );
	BinaryStateMonitor * mem_mon = new BinaryStateMonitor( neurons_out, 0, "mem", sys->fn("bmem") );
	BinaryStateMonitor * vis_mem_mon = new BinaryStateMonitor( neurons_vis, 127, "bg_current", sys->fn("bmemvis") );
	BinaryStateMonitor * g_ampa_mon = new BinaryStateMonitor( neurons_out, 0, "g_ampa", sys->fn("bampa") );
  oss.str("");
	oss << outputfile << "v.w";
	WeightMonitor * smon_wvh = new WeightMonitor( con_ho, oss.str(), auryn_timestep);
  smon_wvh->add_to_list(32,0);
  }

	if ( record_rasters ) {
		msg = "Setting up monitors ...";
		logger->msg(msg,PROGRESS,true);

		filename << outputfile << "err.ras";
		SpikeMonitor * smon_err = new SpikeMonitor( neurons_err, filename.str().c_str() );
		filename.str("");
		filename.clear();
		filename << outputfile << "hid.ras";
		SpikeMonitor * smon_hid = new SpikeMonitor( neurons_hid, filename.str().c_str() );

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
