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

#ifndef ERBPSHAREDCONNECTION_H_
#define ERBPSHAREDCONNECTION_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "peRBPConnection.h"
#include "SimpleMatrix.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>

namespace auryn {
//! \brief <eRBP Conv2D Connection description here>.
//  
//*/
//typedef SimpleMatrix<AurynWeight*> BackwardMatrix;

class peRBPSharedConnection : public peRBPConnection
{
private:
  float* w_kernel;
	void init_kernel(const char * kfilename);
	void free();
  boost::uniform_real<> *dist = new boost::uniform_real<> (0, 1);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> > * die  = new boost::variate_generator<boost::mt19937&, boost::uniform_real<> > ( gen, *dist );

	/*! Controls the strength and sign of the response in the integral controller */

public:
//	bool stdp_active;
//
  int n; //kernel size


	peRBPSharedConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			const char * filename , 
			const char * kfilename , 
      AurynDouble prob_syn,
			TransmitterType transmitter,
      string name = "peRBPSharedConnection"
      );

	peRBPSharedConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			const char * filename , 
			const char * kfilename , 
      AurynDouble prob_syn ,
      AurynDouble learning_rate,
      AurynDouble glow,
      AurynDouble ghigh,
			TransmitterType transmitter,
      string name = "peRBPSharedConnection"
      );

	void propagate_forward();
	void propagate();
	bool write_to_file(string filename);
  //void connect_conv2d_kernel(int kernel_size);

};

}

#endif /*ERBPCONV2DCONNECTION_H_*/
