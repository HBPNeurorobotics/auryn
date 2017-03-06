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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#include "peRBPSharedConnection.h"


using namespace auryn;

void peRBPSharedConnection::init_kernel(const char * kfilename){
	if ( !dst->evolve_locally() ) return; //true

	char buffer[256];
	std::ifstream infile (kfilename);
	if (!infile) {
		std::stringstream oss;
		oss << get_log_name() << "Can't open input file " << kfilename;
		auryn::logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	NeuronID i,j;
	AurynLong k;
	float val;

	// read connection details
	infile.getline (buffer,256);
	std::string header("%%MatrixMarket matrix coordinate real general");
	if (header.compare(buffer)!=0)
	{
		std::stringstream oss;
		oss << get_log_name() << "Input format not recognized.";
		auryn::logger->msg(oss.str(),ERROR);
		return; //false
	}
	while ( buffer[0]=='%' ) {
	  infile.getline (buffer,256);
	}

	sscanf (buffer,"%u %u %lu",&n,&j,&k);
	
  w_kernel = new float[n]; 
  for(int i=0; i<n; i++){
    w_kernel[i]=0;
  }

	std::stringstream oss;
	oss << get_name() 
		<< ": Reading kernel from file";
	auryn::logger->msg(oss.str(),NOTIFICATION);


	while ( infile.getline (buffer,255) )
	{
		sscanf (buffer,"%u %u %e",&i,&j,&val);
    w_kernel[i-1] = val;
	}

	infile.close();

}

peRBPSharedConnection::peRBPSharedConnection(
    SpikingGroup * source,
    NeuronGroup * destination, 
		const char * filename , 
		const char * kfilename , 
    AurynDouble prob_syn,
		TransmitterType transmitter,
    std::string name
    ) 
: peRBPConnection(source, destination, filename, prob_syn, transmitter, name)
{
  init_kernel(kfilename);
  set_name(kfilename);
}

peRBPSharedConnection::peRBPSharedConnection(
    SpikingGroup * source,
    NeuronGroup * destination, 
		const char * filename , 
		const char * kfilename , 
    AurynDouble prob_syn,
    AurynDouble learning_rate,
    AurynDouble gate_lo,
    AurynDouble gate_hi,
		TransmitterType transmitter,
    std::string name
    ) 
: peRBPConnection(source, destination, filename, prob_syn, learning_rate, gate_lo, gate_hi, transmitter, name)
{
  init_kernel(kfilename);
  set_name(kfilename);
}

void peRBPSharedConnection::free()
{
  delete w;
  delete w_kernel;
}


void peRBPSharedConnection::propagate_forward()
{
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != src->get_spikes()->end() ; ++spike ) {
		for (NeuronID * c = w->get_row_begin(*spike) ; c != w->get_row_end(*spike) ; ++c ) { // c = post index
      if ((*die)() < transmission_prob){
			NeuronID * ind = w->get_ind_begin(); // first element of index array
			AurynWeight * data = w->get_data_begin();
			AurynWeight value = w_kernel[(int)data[c-ind]]; 
			transmit( *c , value );
			if ( stdp_active && (sys->get_clock()%2500)>500){
        w_kernel[(int)data[c-ind]] += dw(*c);
      }
      }
		}
	}
}

bool peRBPSharedConnection::write_to_file(string filename){
	if ( !dst->evolve_locally() ) return true;

	std::ofstream outfile;
	outfile.open(filename.c_str(),std::ios::out);
	if (!outfile) {
		std::stringstream oss;
	    oss << "Can't open output file " << filename;
		auryn::logger->msg(oss.str(),ERROR);
		throw AurynOpenFileException();
	}

	outfile << "%%MatrixMarket matrix coordinate real general\n" 
		<< "% Auryn weight matrix. Has to be kept in row major order for load operation.\n" 
		<< "% Connection name: " << get_name() << "\n"
		<< "% Locked range: " << dst->get_locked_range() << "\n"
		<< "%\n"
		<< n << " " << "1" << " " << n << std::endl;

	for ( int i = 0 ; i < n ; ++i ) 
	{
		outfile << std::setprecision(7);
		outfile << i+1 << " " << 1 << " " << std::scientific << w_kernel[i] << std::fixed << "\n";
	}

	outfile.close();
	return true;
}

void peRBPSharedConnection::propagate()
{
	propagate_forward();
}






