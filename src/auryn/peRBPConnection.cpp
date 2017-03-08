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

#include "peRBPConnection.h"

using namespace auryn;

void peRBPConnection::init(AurynDouble learning_rate, AurynDouble prob_syn) 
{
	if ( dst->get_post_size() == 0 ) return;
	// Synaptic traces
  set_eta(learning_rate);
	transmission_prob = prob_syn; // Stochastic blank_out probability
}

peRBPConnection::peRBPConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename , 
    AurynDouble prob_syn,
		TransmitterType transmitter,
    std::string name
    ) 
: SparseConnection(source, destination, filename, transmitter)
{
	init(0., prob_syn);
  modstate = dst->get_state_vector("mem"); 
  vmodstate = dst->get_state_vector("mem"); 
  glow = 0.;
  ghigh = 0.;
  //set_name(name);
}

peRBPConnection::peRBPConnection(
    SpikingGroup * source,
    NeuronGroup * destination, 
		const char * filename , 
    AurynDouble prob_syn,
    AurynDouble learning_rate,
    AurynDouble gate_lo,
    AurynDouble gate_hi,
		TransmitterType transmitter,
    std::string name
    ) 
: SparseConnection(source, destination, filename, transmitter)
{
	init(learning_rate, prob_syn);
  //set_name(name);
  modstate = dst->get_state_vector("dendrite"); 
  vmodstate = dst->get_state_vector("g_ampa"); 
  glow = gate_lo;
  ghigh = gate_hi;
}

void peRBPConnection::free()
{
  delete w;
}

void peRBPConnection::propagate_forward()
{
	for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
			spike != src->get_spikes()->end() ; ++spike ) {
		for (NeuronID * c = w->get_row_begin(*spike) ; c != w->get_row_end(*spike) ; ++c ) { // c = post index
      if ((*die)() < transmission_prob){
			AurynWeight * weight = w->get_data_ptr(c);       
			if ( stdp_active && (sys->get_clock()%2500)>500)  *weight += dw(*c);
      transmit( *c , *weight );

			//NeuronID * ind = w->get_ind_begin(); // first element of index array
			//AurynWeight * weight = w->get_data_begin();
			//AurynWeight value = weight[c-ind]; 
			//transmit( *c , value );
			//if ( stdp_active && (sys->get_clock()%2500)>500)  weight[c-ind] += dw(*c);

      }
		}
	}
}

AurynWeight peRBPConnection::dw(NeuronID post)
{
	NeuronID translated_spike = dst->global2rank(post); 
  if ( vmodstate->get(translated_spike)<ghigh)
    if ( vmodstate->get(translated_spike)>glow){
      return eta*modstate->get(translated_spike);
    }
  return 0.;
}

void peRBPConnection::propagate()
{
	propagate_forward();
}

void peRBPConnection::evolve()
{
	// compute the averages
}

void peRBPConnection::set_eta(AurynFloat value)
{
	eta = value;
  if (eta!=0){
    stdp_active=true;
  } else {
    stdp_active=false;
  }
}





