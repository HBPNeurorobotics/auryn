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

#ifndef PERBPDCONNECTION_H_
#define PERBPDCONNECTION_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "SparseConnection.h"
#include "SimpleMatrix.h"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>


namespace auryn {
//! \brief <peRBP Conv2D Connection description here>.
//  
//*/
class peRBPConnection : public SparseConnection
{
private:
	void init(AurynDouble learning_rate, AurynDouble prob_syn);
	void free();
  boost::uniform_real<> *dist = new boost::uniform_real<> (0, 1);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> > * die  = new boost::variate_generator<boost::mt19937&, boost::uniform_real<> > ( gen, *dist );

	/*! Controls the strength and sign of the response in the integral controller */
public:
//	bool stdp_active;
//
	AurynDouble eta;
  bool stdp_active;
  AurynDouble transmission_prob;
  AurynDouble ghigh;
  AurynDouble glow;
  AurynStateVector * modstate;
  AurynStateVector * vmodstate;
  boost::mt19937 gen = boost::mt19937();

	peRBPConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			const char * filename , 
      AurynDouble blank_out_prob,
			TransmitterType transmitter,
      string name = "peRBPConnection"
      );

	peRBPConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			const char * filename , 
      AurynDouble blank_out_prob,
      AurynDouble learning_rate,
      AurynDouble glow,
      AurynDouble ghigh,
			TransmitterType transmitter,
      string name = "peRBPConnection"
      );

//
//	void set_eta(AurynFloat value);
//
	void propagate_forward();
	void propagate();
	AurynWeight dw(NeuronID post);
	void evolve();
  void set_eta(float learning_rate);
  //void connect_conv2d_kernel(int kernel_size);

};

}

#endif /*ERBPCONV2DCONNECTION_H_*/
