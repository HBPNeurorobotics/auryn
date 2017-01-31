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

#ifndef modulatedVmPlasticCONNECTION_H_
#define modulatedVmPlasticCONNECTION_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "DuplexConnection.h"
#include "Trace.h"
#include "LinearTrace.h"
#include "SpikeDelay.h"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>





namespace auryn {


/*! \brief Double modulatedVmPlastic All-to-All Connection
 *
 * This class implements standard modulatedVmPlastic with a double exponential window and optinal
 * offset terms. Window amplitudes and time constants are freely configurable.
 */
class modulatedVmPlasticConnection : public DuplexConnection
{

private:
	void init(AurynFloat eta, AurynFloat tau_pre, AurynFloat tau_post, AurynFloat maxweight, double prob_syn, double gate_low, double gate_high);

protected:

	AurynDouble hom_fudge;

	Trace * tr_pre;
	Trace * tr_post;

	void propagate_forward();
	void propagate_backward();

	AurynWeight dw_pre(NeuronID post);
	AurynWeight dw_post(NeuronID pre);

public:
	AurynFloat A; /*!< Amplitude of post-pre part of the modulatedVmPlastic window */
  AurynStateVector * modstate;
  AurynStateVector * vmodstate;
  double transmission_prob;
  double glow;
  double ghigh;

	bool stdp_active;

	modulatedVmPlasticConnection(SpikingGroup * source, NeuronGroup * destination, 
			TransmitterType transmitter=GLUT, double prob_syn = 1.0);

	modulatedVmPlasticConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynFloat eta=1, 
			AurynFloat tau_pre=20e-3,
			AurynFloat tau_post=20e-3,
			AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT, 
      double prob_syn = 1.0,
      double glow = -25,
      double ghigh = 25);

	modulatedVmPlasticConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, 
      AurynFloat sparseness=0.05, 
			AurynFloat eta=1, 
			AurynFloat tau_pre=20e-3,
			AurynFloat tau_post=20e-3,
			AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT,
			string name = "modulatedVmPlasticConnection",  
      double prob_syn = 1.0,
      double gate_low = -25,
      double gate_high = 25);


	virtual ~modulatedVmPlasticConnection();
	virtual void finalize();
	void free();

	virtual void propagate();
	virtual void evolve();

};

}

#endif /*modulatedVmPlasticCONNECTION_H_*/
