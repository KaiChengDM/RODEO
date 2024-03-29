/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
 *
 * This file is part of RoDeO
 *
 * RoDeO is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * RoDeO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, TU Kaiserslautern)
 *
 *
 *
 */
#ifndef CONSTRAINT_FUNCTIONS_HPP
#define CONSTRAINT_FUNCTIONS_HPP


#include <armadillo>
#include "objective_function.hpp"
#include "kriging_training.hpp"
#include "aggregation_model.hpp"
#include "design.hpp"
#include <cassert>



class ConstraintDefinition{

public:

	std::string inequalityType;
	std::string name;

	std::string executableName;
	std::string path;
	std::string designVectorFilename;
	std::string outputValueFilename;
	std::string outputGradFilename;

	std::string marker;
	std::string markerForGradient;
	std::string surrogatetype;
	std::string jsonFile;

 /* These are required only for multi-level option */

	std::string executableNameLowFi;
	std::string pathLowFi;
	std::string outputFilenameLowFi;
	std::string markerLowFi;
	std::string markerForGradientLowFi;


	bool ifMultiLevel = false;


	int ID = -1;
    double value = 0;
	int ID_vector_constraint=-1;
	int ID_normal_constraint=-1;
	int ID_gradient_constraint=-1;
	int ID_nongradient_constraint=-1;

	bool ifGradient = false;
	bool ifGradientLowFi = false;
	bool ifDefined = false;
    bool if_vector_constraint = false;   // Created by Kai. For vector field constraint, e.g., f(t,x) > 0, for all t in [0,T].

	ConstraintDefinition(std::string, std::string, double);
	ConstraintDefinition(std::string);
	ConstraintDefinition();
	void print(void) const;

};




class ConstraintFunction: public ObjectiveFunction {


private:


	int ID_vector_constraint=-1;
	int ID_normal_constraint=-1;
	int ID_gradient_constraint=-1;
	int ID_nongradient_constraint=-1;

	//double value;

	bool ifRunNecessary = true;

	int ID = -1;

	bool ifInequalityConstraintSpecified  = false;

	void readOutputWithoutMarkers(Design &d) const;
	void readOutputWithMarkers(Design &d) const;

	bool isInequalityTypeValid(const std::string &);


public:


	ConstraintFunction(std::string, unsigned int);
	ConstraintFunction(std::string, double (*objFun)(double *), unsigned int);

	void readEvaluateOutput(Design &d);
	bool checkFeasibility(double value) const;
	void addDesignToData(Design &);
   // void podROM();                 // proper orthogonal decomposition for dimension reduction

	void setInequalityConstraint(ConstraintDefinition inequalityConstraint);
	void setParametersByDefinition(ConstraintDefinition inequalityConstraint);

	bool checkIfRunExecutableNecessary(void);

	void evaluate(Design &d);

	void setID(int givenID) {
		assert(givenID>=0);
		ID = givenID;

	}

	int getID(void) const {
		return ID;
	}


	void setVectorConstraintID(int ID) {
		 ID_vector_constraint = ID;
	}


	void setNormalConstraintID(int ID) {
	     ID_normal_constraint= ID;
	}

	void setGradConstraintID(int ID) {

	    ID_gradient_constraint = ID;
	}

	void setNonGradConstraintID(int ID){
	    ID_nongradient_constraint = ID;
	}

	int getVectorConstraintID(void) const {
		return ID_vector_constraint;
	}

	int getNormalConstraintID(void) const {
		return ID_normal_constraint;
	}

	int getGradConstraintID(void) const {

		return ID_gradient_constraint;
	}

	int getNonGradConstraintID(void) const {

		return ID_nongradient_constraint;
	}


	int getVectorConstraintLength(int ID) {
		return constraint_length;
	}

	void setrunOn(void) {

		ifRunNecessary = true;
	}

	void setrunOff(void) {

		ifRunNecessary = false;

	}



	void print(void) const;

	double getValue(void) const;
	std::string getInequalityType(void) const;


};



#endif
