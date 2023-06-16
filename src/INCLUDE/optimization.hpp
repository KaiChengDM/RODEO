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
#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP


#include <armadillo>
#include "kriging_training.hpp"
#include "aggregation_model.hpp"
#include "objective_function.hpp"
#include "constraint_functions.hpp"
#include "random_functions.hpp"

class Optimizer {

 private:

//protected:

	vec lowerBounds;
	vec upperBounds;

	vec lowerBoundsForEIMaximization;
	vec upperBoundsForEIMaximization;

	std::string designVectorFileName;

	const std::string optimizationHistoryFileName = "optimizationHistory.csv";
	const std::string globalOptimumDesignFileName = "globalOptimumDesign";

	mat optimizationHistory;

	std::vector<ConstraintFunction> constraintFunctions;

	ObjectiveFunction objFun;

	std::vector<CDesignExpectedImprovement> theMostPromisingDesigns;

	bool ifObjectFunctionIsSpecied = false;
	bool ifSurrogatesAreInitialized = false;
	bool isHistoryFileInitialized = false;

	bool IfinitialValueForObjFunIsSet= false;

	char* workpath;

	Design globalOptimalDesign;

	double initialobjectiveFunctionValue = 0.0;

	double zoomInFactor = 0.5;

	unsigned int iterMaxEILoop;

	// hooke jeeves algorithm

	 int num =1 ;

	 vec dv_lb;
	 vec dv_ub;
	 vec dv_in;

	 mat dv_cur;
	 vec dv_par;
	 vec dv_optimal;

	 vec  increment;
	 uvec ind_increment;

	 mat hyperoptimizationHistory;

	 unsigned int numberOfIteration;

	 vec obj_cur;
	 double obj_optimal;
	 double obj;

public:

	std::string name;

    unsigned int dimension = 0;
	unsigned int numberOfConstraints = 0;
	unsigned int maxNumberOfSamples = 0;
	unsigned int howOftenTrainModels = 10; /* train surrogates in every 10 iteration */

	std::string modelType; // Modified by Kai

	unsigned int sampleDim;

	unsigned int iterGradientEILoop = 100;
	std::string optimizationType = "minimize";

	bool ifVisualize = false;
	bool ifDisplay = false;
	bool ifBoxConstraintsSet = false;

	Optimizer();

	Optimizer(std::string ,int, std::string = "minimize");

	bool checkSettings(void) const;
	void print(void) const;
	void printConstraints(void) const;
	void visualizeOptimizationHistory(void) const;
	void EfficientGlobalOptimization(void);

	void initializeSurrogates(void);
	void trainSurrogates(void);

	void performDoE(unsigned int howManySamples, DoE_METHOD methodID);
	void cleanDoEFiles(void) const;
	void setProblemType(std::string);
	void setMaximumNumberOfIterations(unsigned int );
	void setMaximumNumberOfIterationsForEIMaximization(unsigned int);

	void setBoxConstraints(std::string filename="BoxConstraints.csv");
	void setBoxConstraints(double lb, double ub);
	void setBoxConstraints(vec lb, vec ub);
	void setFileNameDesignVector(std::string filename);

	void setDisplayOn(void);
	void setDisplayOff(void);


	void zoomInDesignSpace(void);

	void setInitialObjectiveFunctionValue(double);
	void calculateImprovementValue(Design &);

	void addConstraint(ConstraintFunction &);

	void evaluateConstraints(Design &);
	void addConstraintValuesToDoEData(Design &) const;


	void estimateConstraints(CDesignExpectedImprovement &) const;

	void checkIfSettingsAreOK(void) const;
	bool checkBoxConstraints(void) const;
	bool checkConstraintFeasibility(rowvec) const;

	void addObjectFunction(ObjectiveFunction &);


	void addPenaltyToExpectedImprovementForConstraints(CDesignExpectedImprovement &) const;

	void computeConstraintsandPenaltyTerm(Design &);


	void findTheGlobalOptimalDesign(void);


	void updateOptimizationHistory(Design d);
	void clearOptimizationHistoryFile(void) const;
	void prepareOptimizationHistoryFile(void) const;


	void addConstraintValuesToData(Design &d);

	double Evaluate (vec dv);          // compute the constrained EI function

	rowvec calculateEIGradient(CDesignExpectedImprovement &) const;

	CDesignExpectedImprovement MaximizeEIGradientBased(CDesignExpectedImprovement ) const;

    CDesignExpectedImprovement local_search(CDesignExpectedImprovement &);  // use hooke-jeeves algorithm for local search

	void findTheMostPromisingDesign(unsigned int howManyDesigns = 1);

	CDesignExpectedImprovement getDesignWithMaxExpectedImprovement(void) const;

	rowvec generateRandomRowVectorAroundASample(void);


	bool ifConstrained(void) const;

	void displayMessage(std::string) const;


   // hooke jeeves algorithm

	void boxmin(mat dv, vec dv_lb, vec dv_ub);
	void start(vec dv_in, vec dv_lb, vec dv_ub, int num);
	void explore(vec dv_in, double obj, int num);
	void move(vec dv_1, vec dv_2, double obj, int num);

	mat get_dv(void) ;
    vec get_obj(void) ;
	vec getOptimal_dv(void) ;
	double getOptimal_obj(void);

};


#endif
