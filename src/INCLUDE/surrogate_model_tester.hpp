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

#ifndef SURROGATE_MODEL_TESTER_HPP
#define SURROGATE_MODEL_TESTER_HPP


#include <string>
#include "output.hpp"
#include "bounds.hpp"
#include "linear_regression.hpp"
#include "Rodeo_macros.hpp"
#include "kriging_training.hpp"
#include "aggregation_model.hpp"
#include "multi_level_method.hpp"
#include "gek.hpp"     // Modified by Kai
#include "sgek.hpp"    // Modified by Kai

class SurrogateModelTester{


private:


	string name;

	unsigned int output_dimension=1;
	unsigned int dimension = 0;

	OutputDevice outputToScreen;
	Bounds boxConstraints;

	LinearModel linearModel;
	KrigingModel krigingModel;
	AggregationModel aggregationModel;
	MultiLevelModel multilevelModel;
	GEKModel gekmodel;        // Modified by Kai
	SGEKModel sgekmodel;      // Modified by Kai

	SurrogateModel *surrogateModel;

	SURROGATE_MODEL surrogateModelType;

	unsigned int numberOfTrainingIterations = 10000;

	bool ifSurrogateModelSpecified = false;
	bool ifVectorOutput = false;

	string fileNameTraingData;
	string fileNameTraingDataLowFidelity;
	string fileNameTestData;

public:

	SurrogateModelTester();

	unsigned int getDimension(void) const;

	void setName(string);

	void setDimension(unsigned int);
	void setOutputVectorDimension(unsigned int);

	void setDisplayOn(void);
	void setDisplayOff(void);

	void setNumberOfTrainingIterations(unsigned int);

	void setSurrogateModel(SURROGATE_MODEL);
	bool isSurrogateModelSpecified(void) const;


	void setBoxConstraints(Bounds);
	Bounds getBoxConstraints(void) const;

	void performSurrogateModelTest(void);

	void setFileNameTrainingData(string);
	void setFileNameTrainingDataLowFidelity(string);


	void setFileNameTestData(string);

	void print(void) const;
};

#endif
