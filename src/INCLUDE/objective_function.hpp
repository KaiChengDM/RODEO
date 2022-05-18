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
#ifndef OBJECTIVE_FUNCTION_HPP
#define OBJECTIVE_FUNCTION_HPP

#include <fstream>
#include <armadillo>
#include "kriging_training.hpp"
#include "aggregation_model.hpp"
#include "surrogate_model.hpp"
#include "multi_level_method.hpp"
#include "design.hpp"
#include "output.hpp"
#include "sgek.hpp"
#include "gek.hpp"
#include "Rodeo_macros.hpp"

class ObjectiveFunctionDefinition{

public:
	std::string name;
	std::string designVectorFilename;

	std::string executableName;
	std::string path;
	std::string outputFilename;
	std::string marker;
	std::string markerForGradient;
	std::string surrogatetype;  // Modified by Kai


	/* These are required only for multi-level option */
	std::string executableNameLowFi;
	std::string pathLowFi;
	std::string outputFilenameLowFi;
	std::string markerLowFi;
	std::string markerForGradientLowFi;



	bool ifMultiLevel = false;
	bool ifGradient = false;
	bool ifGradientLowFi = false;
	bool ifDefined = false;

	ObjectiveFunctionDefinition(std::string);
	ObjectiveFunctionDefinition();
	void print(void) const;


};


class ObjectiveFunction{


private:



protected:


	double (*objectiveFunPtr)(double *);
	double (*objectiveFunAdjPtr)(double *,double *);

	std::string name;
	std::string fileNameDesignVector;


	std::string executableName;
	std::string executablePath;
	std::string fileNameInputRead;
	std::string readMarker;
	std::string readMarkerAdjoint;

	std::string executableNameLowFi;
	std::string executablePathLowFi;
	std::string fileNameInputReadLowFi;
	std::string readMarkerLowFi;
	std::string readMarkerAdjointLowFi;
	std::string surrogatetype;  // Modified by Kai
	std::string fileNameTrainingDataForSurrogate;


	bool ifMarkerIsSet = false;
	bool ifAdjointMarkerIsSet = false;


	vec upperBounds;
	vec lowerBounds;


	KrigingModel surrogateModel;
	AggregationModel surrogateModelGradient;
	MultiLevelModel surrogateModelML;

	GEKModel gekModel;     // Modified by Kai
	SGEKModel sgekModel;   // Modified by Kai

	SurrogateModel *surrogate;

	OutputDevice output;

	unsigned int numberOfIterationsForSurrogateTraining = 10000;


	unsigned int dim = 0;
	bool ifDoErequired = true;
	bool ifWarmStart = false;
	bool ifGradientAvailable = false;
	bool ifFunctionPointerIsSet = false;
	bool ifInitialized = false;
	bool ifParameterBoundsAreSet = false;
	bool ifMultilevel = false;
	bool ifDefinitionIsSet = false;


	void readOutputWithoutMarkers(Design &outputDesignBuffer) const;

	bool checkIfMarkersAreNotSet(void) const;

public:

	ObjectiveFunction(std::string, unsigned int);
	ObjectiveFunction();

	void bindSurrogateModel(void);
	//void bindSurrogateModel(SURROGATE_MODEL); // Modified by Kai

	void setFunctionPointer(double (*objFun)(double *));
	void setFunctionPointer(double (*objFun)(double *, double *));

	void initializeSurrogate(void);
	void trainSurrogate(void);
	void printSurrogate(void) const;

	KrigingModel     getSurrogateModel(void) const;
	AggregationModel getSurrogateModelGradient(void) const;
	MultiLevelModel  getSurrogateModelML(void) const;
	GEKModel      getGEKModel(void) const;  // Modified by Kai
	SGEKModel     getSGEKModel(void) const; // Modified by Kai

	SURROGATE_MODEL surrogateModelType;  // Modified by Kai

	void setGradientOn(void);
	void setGradientOff(void);

	void setDisplayOn(void);
	void setDisplayOff(void);


	void setParameterBounds(vec , vec );

	void setNumberOfTrainingIterationsForSurrogateModel(unsigned int);

	unsigned int getDimension(void) const{

		return dim;
	}

	std::string getName(void) const{

		return name;
	}

	bool ifHasFunctionFunctionPointer(void) const{

		return ifFunctionPointerIsSet;

	}

	void setFileNameReadInput(std::string fileName);

	void saveDoEData(std::vector<rowvec>) const;
	void setExecutablePath(std::string);
	void setExecutableName(std::string);
	void setFileNameDesignVector(std::string);


	void setReadMarker(std::string marker);
	std::string getReadMarker(void) const;

	size_t isMarkerFound(const std::string &marker, const std::string &inputStr) const;

	double getMarkerValue(const std::string &inputStr, size_t foundMarker) const;
	rowvec getMarkerAdjointValues(const std::string &inputStr, size_t foundMarkerPosition) const;

	void setReadMarkerAdjoint(std::string marker);
	std::string getReadMarkerAdjoint(void) const;


	void setParametersByDefinition(ObjectiveFunctionDefinition);


	void calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const;

	void evaluate(Design &d);
	void evaluateAdjoint(Design &d);
	void readEvaluateOutput(Design &d);

	void addDesignToData(Design &d);

	bool checkIfGradientAvailable(void) const;
	double interpolate(rowvec x) const;
	void print(void) const;
	std::string getExecutionCommand(void) const;
	std::string getExecutionCommandLowFi(void) const;


};


#endif
