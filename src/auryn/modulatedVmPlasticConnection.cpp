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

#include "modulatedVmPlasticConnection.h"

boost::mt19937 gen = boost::mt19937();

boost::uniform_real<> *dist = new boost::uniform_real<> (0, 1);

boost::variate_generator<boost::mt19937&, boost::uniform_real<> > * die  = new boost::variate_generator<boost::mt19937&, boost::uniform_real<> > ( gen, *dist );


using namespace auryn;

void modulatedVmPlasticConnection::init(AurynFloat eta, AurynFloat tau_pre, AurynFloat tau_post, AurynFloat maxweight, double prob_syn, double gate_low, double gate_high)
{
	if ( dst->get_post_size() == 0 ) return;

	A = eta; // post-pre

	auryn::logger->parameter("eta",eta);
  glow = gate_low;
  ghigh = gate_high;
	tr_pre  = src->get_pre_trace(tau_pre);
	tr_post = dst->get_post_trace(tau_post);

	set_min_weight(-maxweight);
	set_max_weight(maxweight);
  transmission_prob = prob_syn;
  printf("prob_syn %f", transmission_prob);

	stdp_active = true;
  modstate = dst->get_state_vector("dendrite");
  vmodstate = dst->get_state_vector("g_ampa");
}


void modulatedVmPlasticConnection::finalize() {
	DuplexConnection::finalize();
}

void modulatedVmPlasticConnection::free()
{
}

modulatedVmPlasticConnection::modulatedVmPlasticConnection(
    SpikingGroup * source, NeuronGroup * destination,
    TransmitterType transmitter,
    double prob_syn) : DuplexConnection(source, destination, transmitter)
{
}

modulatedVmPlasticConnection::modulatedVmPlasticConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
		AurynFloat eta,
		AurynFloat tau_pre,
		AurynFloat tau_post,
		AurynFloat maxweight, 
		TransmitterType transmitter,
    double  prob_syn,
    double gate_low,
    double gate_high) 
: DuplexConnection(source, 
		destination, 
		filename, 
		transmitter)
{
	init(eta, tau_pre, tau_post, maxweight, prob_syn, gate_low, gate_high);
}

modulatedVmPlasticConnection::modulatedVmPlasticConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynFloat eta, 
		AurynFloat tau_pre,
		AurynFloat tau_post,
		AurynFloat maxweight, 
		TransmitterType transmitter,
		std::string name, 
    double  prob_syn,
    double gate_low,
    double gate_high) 
: DuplexConnection(source, 
		destination, 
		weight, 
		sparseness, 
		transmitter, 
		name)
{
	init(eta, tau_pre, tau_post, maxweight, prob_syn, gate_low, gate_high);
	if ( name.empty() )
		set_name("modulatedVmPlasticConnection");
}

modulatedVmPlasticConnection::~modulatedVmPlasticConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}


AurynWeight modulatedVmPlasticConnection::dw_pre(NeuronID post)
{
	NeuronID translated_spike = dst->global2rank(post); 
  if ( vmodstate->get(translated_spike)<ghigh){
    if (vmodstate->get(translated_spike)>glow) {
	AurynDouble dw = A;
  dw *= modstate->get(translated_spike);
	return dw;
  }
  }
  return 0.;
}

AurynWeight modulatedVmPlasticConnection::dw_post(NeuronID pre)
{
	return 0.;
}


void modulatedVmPlasticConnection::propagate_forward()
{
	// loop over all spikes
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != src->get_spikes()->end() ; ++spike ) {
		// loop over all postsynaptic partners
		for (const NeuronID * c = w->get_row_begin(*spike) ; 
				c != w->get_row_end(*spike) ; 
				++c ) { // c = post index

			// transmit signal to target at postsynaptic neuron
			AurynWeight * weight = w->get_data_ptr(c);       
      if (transmission_prob==1.){
        transmit( *c , *weight );
			if ( stdp_active && (sys->get_clock()%2500)>500)  *weight += dw_pre(*c);
			// if ( stdp_active )  *weight += dw_pre(*c);

      }else{
      if ((*die)() < transmission_prob){
            transmit( *c , *weight );
      }
			if ( stdp_active && (sys->get_clock()%2500)>500) *weight += dw_pre(*c);
			//if ( stdp_active ) *weight += dw_pre(*c);
      }

			// handle plasticity

		}
	}
}

void modulatedVmPlasticConnection::propagate_backward()
{
}

void modulatedVmPlasticConnection::propagate()
{
	propagate_forward();
}

void modulatedVmPlasticConnection::evolve()
{
}

