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

#ifndef LINGROUP_H_
#define LINGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "System.h"

namespace auryn {

/*! \brief Conductance based LIF neuron model with absolute refractoriness as used in Vogels and Abbott 2005.
 */
class LinGroup : public NeuronGroup
{
private:
	auryn_vector_float * bg_current;
  AurynStateVector * temp;
	AurynFloat scale_mem;

	void init();
  AurynFloat thr;
	inline void integrate_state();
	inline void check_thresholds();
	virtual string get_output_line(NeuronID i);
	virtual void load_input_line(NeuronID i, const char * buf);

	void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version );
	void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version );
public:
	AurynInt * spike_count; 
	/*! \brief The default constructor of this NeuronGroup */
	LinGroup(NeuronID size);
	virtual ~LinGroup();

	/*! \brief Controls the constant current input (per default set so zero) to neuron i */
	void set_bg_current(NeuronID i, AurynFloat current);

	/*! \brief Controls the constant current input to all neurons */
	void set_bg_currents(AurynFloat current);

	/*! \brief Resets all neurons to defined and identical initial state. */
	void clear();

	/*! \brief Integrates the NeuronGroup state
	 *
	 * The evolve method internally used by System. */
	void evolve();
};

}

#endif /*TIFGROUP_H_*/

