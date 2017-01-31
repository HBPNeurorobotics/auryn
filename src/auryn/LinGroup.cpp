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

#include "LinGroup.h"

using namespace auryn;

LinGroup::LinGroup(NeuronID size) : NeuronGroup(size)
{
	auryn::sys->register_spiking_group(this);
	if ( evolve_locally() ) init();
}

void LinGroup::init()
{
	thr = 100e-3;

	bg_current = get_state_vector("bg_current");
  temp = get_state_vector("_temp");

	clear();

}

void LinGroup::clear()
{
	clear_spikes();
  spike_count = new AurynInt[get_rank_size()];
	for (NeuronID i = 0; i < get_rank_size(); i++) {
	   auryn_vector_float_set (mem, i, 0);
	   auryn_vector_float_set (bg_current, i, 0.);
	   spike_count[i] = 0;     
	}
}


LinGroup::~LinGroup()
{
	if ( !evolve_locally() ) return;
}


void LinGroup::evolve()
{
  temp->add(bg_current);
  temp->mul(auryn_timestep);
  mem->saxpy(1,temp);
	for (NeuronID i = 0 ; i < get_rank_size() ; ++i ) {

			if (mem->get(i)>thr) {
				push_spike(i);
        spike_count[i]++;
				mem->set(i, mem->get(i) - thr);
			} else if (mem->get(i)<0) {
				mem->set(i, 0);

      }
	}
}

void LinGroup::set_bg_current(NeuronID i, AurynFloat current) {
	if ( localrank(i) )
		auryn_vector_float_set ( bg_current , global2rank(i) , current ) ;
}

void LinGroup::set_bg_currents(AurynFloat current) {
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i ) 
		auryn_vector_float_set ( bg_current , i , current ) ;
}

std::string LinGroup::get_output_line(NeuronID i)
{
	std::stringstream oss;
	oss << rank2global(i) << " " << spike_count[i] << "\n";
	return oss.str();
}

void LinGroup::load_input_line(NeuronID i, const char * buf)
{
		float vmem,vampa,vgaba,vbgcur;
		NeuronID vref;
		sscanf (buf,"%f %f",&vmem,&vbgcur);
		if ( localrank(i) ) {
			NeuronID trans = global2rank(i);
			mem->set(trans,vmem);
			bg_current->set(trans, vbgcur);
		}
}


void LinGroup::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	SpikingGroup::virtual_serialize(ar,version);
}

void LinGroup::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	SpikingGroup::virtual_serialize(ar,version);
}
