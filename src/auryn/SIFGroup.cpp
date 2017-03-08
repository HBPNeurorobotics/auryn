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

#include "SIFGroup.h"

using namespace auryn;

SIFGroup::SIFGroup(NeuronID size, NodeDistributionMode node_distr_mode) : NeuronGroup(size, node_distr_mode)
{
  mode = node_distr_mode; 
	auryn::sys->register_spiking_group(this);
	if ( evolve_locally() ) init();

}

void SIFGroup::calculate_scale_constants()
{
	scale_mem  = auryn_timestep/tau_mem;
	scale_dendrite  = auryn_timestep/tau_dendrite;
	scale_ampa = exp(-auryn_timestep/tau_ampa);
	scale_gaba = exp(-auryn_timestep/tau_gaba);

}

void SIFGroup::init()
{
	e_rest = 0e-3;
	thr = 100e-3;
	tau_ampa = 4e-3;
	tau_gaba = 4e-3;
    r_mem = 1e9;
    c_mem = 1e-12;
	tau_mem = r_mem*c_mem;
	tau_dendrite = 5e-3;
	set_refractory_period(4e-3);

	calculate_scale_constants();
	
	ref = auryn_vector_ushort_alloc (get_vector_size()); 
	bg_current = get_state_vector("bg_current");
	bg_current_dendrite = get_state_vector("bg_current_dendrite");

  dendrite = get_state_vector("dendrite");
  t_dendrite = get_state_vector("_dendrite");
  t_mem = get_state_vector("_mem");
	t_ref = auryn_vector_ushort_ptr ( ref , 0 ); 

	clear();

}

void SIFGroup::clear()
{
	clear_spikes();
  spike_count = new AurynInt[get_rank_size()];

	for (NeuronID i = 0; i < get_rank_size(); i++) {
	   auryn_vector_float_set (mem, i, e_rest);
	   auryn_vector_float_set (dendrite, i, e_rest);
	   auryn_vector_ushort_set (ref, i, 0);
	   auryn_vector_float_set (g_ampa, i, 0.);
	   auryn_vector_float_set (g_gaba, i, 0.);
	   auryn_vector_float_set (bg_current, i, 0.);
	   auryn_vector_float_set (bg_current_dendrite, i, 0.);
	   spike_count[i] = 0;     
	}
}


SIFGroup::~SIFGroup()
{
	if ( !evolve_locally() ) return;

	auryn_vector_ushort_free (ref);
}


void SIFGroup::evolve()
{
//t_mem->add(e_rest);
t_mem->sub(mem);
t_mem->add(bg_current);
t_mem->saxpy(scale_ampa,g_ampa);
mem->saxpy(scale_mem,t_mem);


//t_dendrite->add(e_rest);
t_dendrite->sub(dendrite);
t_dendrite->add(bg_current_dendrite);
dendrite->saxpy(scale_dendrite,t_dendrite);

for (NeuronID i = 0 ; i < get_rank_size() ; ++i ) {
  	if (t_ref[i]>0) {
      mem->set(i,e_rest);
      t_ref[i]-- ;
    }

		if (mem->get(i)>thr) {
			push_spike(i);
      spike_count[i]++;
			mem->set(i, e_rest) ;
			t_ref[i] += refractory_time ;
		} 

}

g_ampa->scale(scale_ampa);
t_dendrite->mul(0.);
t_mem->mul(0.);
//g_gaba->scale(scale_gaba);
}

void SIFGroup::set_bg_current(NeuronID i, AurynFloat current) {
	if ( localrank(i) )
		auryn_vector_float_set ( bg_current , global2rank(i) , current ) ;
}

void SIFGroup::set_bg_currents(AurynFloat current) {
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i ) 
		auryn_vector_float_set ( bg_current , i , current ) ;
}

void SIFGroup::set_bg_current_dendrite(NeuronID i, AurynFloat current) {
	if ( localrank(i) )
		auryn_vector_float_set ( bg_current_dendrite , global2rank(i) , current ) ;
}

void SIFGroup::set_bg_currents_dendrite(AurynFloat current) {
	for ( NeuronID i = 0 ; i < get_rank_size() ; ++i ) 
		auryn_vector_float_set ( bg_current_dendrite , i , current ) ;
}

void SIFGroup::set_tau_mem(AurynFloat taum)
{
	tau_mem = taum;
	calculate_scale_constants();
}

void SIFGroup::set_r_mem(AurynFloat rm)
{
	r_mem = rm;
	tau_mem = r_mem*c_mem;
	calculate_scale_constants();
}

void SIFGroup::set_c_mem(AurynFloat cm)
{
	c_mem = cm;
	tau_mem = r_mem*c_mem;
	calculate_scale_constants();
}

void SIFGroup::set_thr(AurynFloat thr_)
{
	thr = thr_;
}

AurynFloat SIFGroup::get_bg_current(NeuronID i) {
	if ( localrank(i) )
		return auryn_vector_float_get ( bg_current , global2rank(i) ) ;
	else 
		return 0;
}

std::string SIFGroup::get_output_line(NeuronID i)
{
	std::stringstream oss;
	oss << rank2global(i) << " " << spike_count[i] << "\n";
	return oss.str();
}

void SIFGroup::load_input_line(NeuronID i, const char * buf)
{
		float vmem,vampa,vgaba,vbgcur;
		NeuronID vref;
		sscanf (buf,"%f %f %f %u %f",&vmem,&vampa,&vgaba,&vref,&vbgcur);
		if ( localrank(i) ) {
			NeuronID trans = global2rank(i);
			mem->set(trans,vmem);
			g_ampa->set(trans,vampa);
			g_gaba->set(trans,vgaba);
			ref->set(trans, vref);
			bg_current->set(trans, vbgcur);
		}
}

void SIFGroup::set_tau_ampa(AurynFloat taum)
{
	tau_ampa = taum;
	calculate_scale_constants();
}

AurynFloat SIFGroup::get_tau_ampa()
{
	return tau_ampa;
}

void SIFGroup::set_tau_gaba(AurynFloat taum)
{
	tau_gaba = taum;
	calculate_scale_constants();
}

AurynFloat SIFGroup::get_tau_gaba()
{
	return tau_gaba;
}

void SIFGroup::set_refractory_period(double t)
{
	refractory_time = (unsigned short) (t/auryn_timestep);
}

void SIFGroup::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	SpikingGroup::virtual_serialize(ar,version);
	ar & *ref;
}

void SIFGroup::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	SpikingGroup::virtual_serialize(ar,version);
	ar & *ref;
}
