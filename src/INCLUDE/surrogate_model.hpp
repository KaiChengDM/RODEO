/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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

#ifndef SURROAGE_MODEL_HPP
#define SURROAGE_MODEL_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"
#include "bounds.hpp"
#include "design.hpp"
#include "surrogate_model_data.hpp"

using namespace arma;
using std::string;


class SurrogateModel{

protected:

	unsigned int NTest = 0;

	vec yTest;
	mat yVectorTest;     // for vector output
	mat XTestraw;
	mat XTest;

	std::string name;

	std::string hyperparameters_filename;

	std::string filenameDataInput;
	std::string filenameDataInputTest;

	std::string filenameTestResults;

	unsigned int numberOfHyperParameters  = 0;
	unsigned int numberOfTrainingIterations  = 10000;

	SurrogateModelData data;

	OutputDevice output;

	bool ifHasGradientData = false;
	bool ifVectorOutput    = false;

public:

	bool ifInitialized = false;
	bool ifDataIsRead = false;
	bool ifNormalized = false;
	bool ifHasTestData = false;
	bool ifDataIsAssigned = false;      // for vector output
	bool ifDimensionReduction = false;  // for dimension reduction

	mat testResults;

	SURROGATE_MODEL modelID;


	SurrogateModel();
	SurrogateModel(std::string name);

	void setName(std::string);

	void readDataTest(void);
	void normalizeDataTest(void);
	virtual void readData(void);

	void assignOutput(int ID);
	int readRank(void) const;
	double readOutputMean(void) const;
	double readOutputStd(void) const;
	rowvec readOutputMeanVector(void) const;
	rowvec readOutputStdVector(void) const;

	mat getPodBasis(void) const;

	virtual void normalizeData(void);

	void checkRawData(void) const;

	void setBoxConstraints(vec xmin, vec xmax);
	void setBoxConstraints(double xmin, double xmax);
	void setBoxConstraints(Bounds boxConstraintsInput);

	void setBoxConstraintsFromData(void);

	void setGradientsOn(void);
	void setGradientsOff(void);
	bool areGradientsOn(void) const;

	void setVectorConstraintOn(void);
	void setVectorConstraintOff(void);
	bool areVectorConstraint(void) const;

	void setConstraintLength(int length);

	virtual void setDisplayOn(void);
	virtual void setDisplayOff(void);

	string getNameOfHyperParametersFile(void) const;
	string getNameOfInputFile(void) const;

	unsigned int getDimension(void) const;
	unsigned int getNumberOfSamples(void) const;
	mat getRawData(void) const;


	void setNameOfInputFileTest(string filename);

	virtual void setNameOfInputFile(string filename) = 0;
	virtual void setNameOfHyperParametersFile(string filename) = 0;
	virtual void setNumberOfTrainingIterations(unsigned int) = 0;

	virtual void initializeSurrogateModel(void) = 0;
	virtual void printSurrogateModel(void) const = 0;
	virtual void printHyperParameters(void) const = 0;
	virtual void saveHyperParameters(void) const = 0;
	virtual void loadHyperParameters(void) = 0;
	virtual void updateAuxilliaryFields(void);
	virtual void train(void) = 0;

	virtual double interpolate(rowvec x) const = 0;
	virtual vec interpolate_vec(rowvec x) const = 0;

	//virtual mat interpolate(void) const = 0;  // Modified by Kai

	virtual void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const = 0;
	virtual void interpolateWithVariance_vec(rowvec xp,vec &f_tilde, vec &ssqr) const = 0;

	virtual void calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const = 0;

	virtual void addNewSampleToData(rowvec newsample) = 0;

	void tryOnTestData(void) const;

	double calculateInSampleError(void) const;
	void calculateOutSampleError(void);
	double getOutSampleErrorMSE(void) const;
	void saveTestResults(void) const;


	rowvec getRowX(unsigned int index) const;
	rowvec getRowXRaw(unsigned int index) const;

	void visualizeTestResults(void) const;

	/* Hooke Jeeves algorithm  // Created by Kai

	void boxmin(vec hyper_lb, vec hyper_ub, int num);
	void start(vec int_hyper, vec hyper_lb, vec hyper_ub, int num);
	void explore(vec int_hyper, double likelihood, int num);
	void move(vec hyper_1, vec hyper_2, double likelihood, int num);

	mat getTheta(void) const;
	vec getLikelihood(void) const;
	vec getOptimalTheta(void) const;
	double getOptimalLikelihood(void) const;*/

};


#endif
