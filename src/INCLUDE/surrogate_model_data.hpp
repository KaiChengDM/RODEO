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

#ifndef SURROGATE_MODEL_DATA_HPP
#define SURROGATE_MODEL_DATA_HPP

#include "output.hpp"
#include "bounds.hpp"
#include<string>

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;
using std::string;

class SurrogateModelData{

private:

	unsigned int numberOfSamples = 0;
	unsigned int numberOfTestSamples = 0;
	unsigned int dimension = 0;
	unsigned int constraintLength = 0;

	bool ifDataHasGradients = false;
	bool ifDataIsNormalized = false;
	bool ifVectorOutput = false;   // for vector output
	bool ifOutputIsNormalized= false;

	mat rawData;
	mat X;
	mat Xraw;

	mat gradient;

	vec y;
    mat y_vec;                     // for vector output

    double mean_y = 0;
	double std_y = 1;

	rowvec mean_basiscoefficient;
	rowvec std_basiscoefficient;

    mat pod_basis;
    mat pod_basiscoefficient;
    vec eigenvalue;

    double threshold =  0.99999;
    int rank;

	mat XrawTest;
	mat XTest;

    vec YTest;
    mat Y_vectTest;               // for vector outputY

	OutputDevice outputToScreen;
	Bounds boxConstraints;


public:

	SurrogateModelData();


	void setDisplayOn(void);
	void setDisplayOff(void);

	void setGradientsOn(void);
	void setGradientsOff(void);

	void setVectorOutputOn(void);   // created by kai
	void setVectorOutputOff(void);  // created by kai

	void setConstraintLength(int length);   // created by kai
	int  getConstraintLength(void) const;    // created by kai

	void setBoxConstraints(Bounds);
	void setBoxConstraintsFromData(void);

	void pod_ROM(void);
	void assignOutput(int ID);
	int getRank(void) const;

	Bounds getBoxConstraints(void) const;


	void assignDimensionFromData(void);
	void assignSampleInputMatrix(void);

	void assignSampleOutputVector(void);
	void assignSampleOutputMatrix(void);

	void assignGradientMatrix(void);

	void normalizeSampleInputMatrix(void);
	void normalizeSampleOutput(void);

	void normalizeSampleInputMatrixTest(void);
	// void normalizeSampleOutputTest(void);


	bool isDataNormalized(void) const;

	bool isVectorOutput(void) const;

	unsigned int getNumberOfSamples(void) const;
	unsigned int getNumberOfSamplesTest(void) const;

	unsigned int getDimension(void) const;
	void setDimension(unsigned int);

	mat getRawData(void) const;

	rowvec getRowX(unsigned int index) const;
	rowvec getRowXTest(unsigned int index) const;
	rowvec getRowRawData(unsigned int index) const;
	rowvec getRowGradient(unsigned int index) const;
	vec getYTest(void) const;
	mat getY_VectorTest(void) const;


	mat getInputMatrix(void) const;

	rowvec getRowXRaw(unsigned int index) const;
	rowvec getRowXRawTest(unsigned int index) const;

	vec getOutputVector(void) const;

	double getOutputMean(void) const;
	double getOutputStd(void) const;

	rowvec getOutputMeanVector(void) const;
	rowvec getOutputStdVector(void) const;

	mat getPodBasis(void) const;
	mat getPodBasisCoefficients(void) const;

	double getMinimumOutputVector(void) const;

	mat getGradientMatrix(void) const;

	void readData(string);

	void readDataTest(string);


	void print(void) const;



};



#endif
