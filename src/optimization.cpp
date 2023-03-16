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
 * Authors: Emre Özkaya, (SciComp, TU Kaiserslautern), Kai Cheng (SDU)
 *
 *
 *
 */

#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>
#include "auxiliary_functions.hpp"
#include "kriging_training.hpp"
#include "aggregation_model.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "test_functions.hpp"
#include "optimization.hpp"
#include "lhs.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


using namespace arma;


Optimizer::Optimizer(std::string nameTestcase, int numberOfOptimizationParams, std::string problemType){

	/* RoDeO does not allow problems with too many optimization parameters */

	if(numberOfOptimizationParams > 100){

		std::cout<<"ERROR: Problem dimension of the optimization is too large!"<<std::endl;
		abort();

	}

	name = nameTestcase;
	dimension = numberOfOptimizationParams;
	sampleDim = dimension;

	lowerBounds.zeros(dimension);
	upperBounds.zeros(dimension);

	lowerBoundsForEIMaximization.zeros(dimension);
	upperBoundsForEIMaximization.zeros(dimension);
	upperBoundsForEIMaximization.fill(1.0/dimension);


	iterMaxEILoop = dimension*100000;


	setProblemType(problemType);



}

bool Optimizer::checkSettings(void) const{

	if(this->ifDisplay){

		std::cout<<"Checking settings...\n";

	}


	bool ifAllSettingsOk = true;

	if(!ifBoxConstraintsSet){

		ifAllSettingsOk = false;

	}

	return ifAllSettingsOk;
}


void Optimizer::setProblemType(std::string type){

	assert(!type.empty());

	if(type == "MAXIMIZATION" || type == "Maximize" || type == "maximization" || type == "maximize" ){

		type = "maximize";

	}

	else if(type == "MINIMIZATION" || type == "Minimize" || type == "minimization" || type == "minimize"){

		type = "minimize";

	}

	else if(type == "DOE" || type == "doe" || type == "DoE" || type == "Doe"){

		type = "DoE";

	}

	else{

		std::cout<<"ERROR: Problem type is undefined!\n";
		abort();

	}


	optimizationType = type;



}

void Optimizer::setInitialObjectiveFunctionValue(double value){

	initialobjectiveFunctionValue = value;
	IfinitialValueForObjFunIsSet = true;

}

void Optimizer::setMaximumNumberOfIterations(unsigned int maxIterations){

	maxNumberOfSamples = maxIterations;

}

void Optimizer::setMaximumNumberOfIterationsForEIMaximization(unsigned int maxIterations){

	iterMaxEILoop = maxIterations;

}


void Optimizer::setFileNameDesignVector(std::string filename){

	assert(!filename.empty());
	designVectorFileName = filename;

}


void Optimizer::setBoxConstraints(std::string filename){

	assert(!filename.empty());

	if(this->ifDisplay){

		std::cout<<"Setting box constraints for "<<name<<std::endl;

	}

	mat boxConstraints;

	bool status = boxConstraints.load(filename.c_str(), csv_ascii);
	if(status == true)
	{
		std::cout<<"Input for the box constraints is done"<<std::endl;
	}
	else
	{
		std::cout<<"Problem with data the input (cvs ascii format) at"<<__FILE__<<", line:"<<__LINE__<<std::endl;
		exit(-1);
	}

	for(unsigned int i=0; i<dimension; i++){

		assert(boxConstraints(i,0) < boxConstraints(i,1));

	}

	lowerBounds = boxConstraints.col(0);
	upperBounds = boxConstraints.col(1);
	ifBoxConstraintsSet = true;


}

void Optimizer::setBoxConstraints(double lowerBound, double upperBound){

	assert(lowerBound < upperBound);
	if(this->ifDisplay){

		std::cout<<"Setting box constraints for "<<name<<std::endl;
	}

	assert(lowerBound < upperBound);
	lowerBounds.fill(lowerBound);
	upperBounds.fill(upperBound);
	ifBoxConstraintsSet = true;

}


void Optimizer::setBoxConstraints(vec lb, vec ub){

	assert(lb.size()>0);
	assert(lb.size() == ub.size());
	assert(ifBoxConstraintsSet == false);

	if(this->ifDisplay){

		std::cout<<"Setting box constraints for "<<name<<std::endl;

	}
	for(unsigned int i=0; i<dimension; i++) assert(lb(i) < ub(i));

	lowerBounds = lb;
	upperBounds = ub;
	ifBoxConstraintsSet = true;

}


void Optimizer::setDisplayOn(void){

	ifDisplay = true;

}
void Optimizer::setDisplayOff(void){

	ifDisplay = false;
}


void Optimizer::addConstraint(ConstraintFunction &constFunc){

	constraintFunctions.push_back(constFunc);
	numberOfConstraints++;
	sampleDim++;

}


void Optimizer::addObjectFunction(ObjectiveFunction &objFunc){

	assert(ifObjectFunctionIsSpecied == false);
	objFun = objFunc;
	sampleDim++;
	ifObjectFunctionIsSpecied = true;

}


void Optimizer::evaluateConstraints(Design &d){


	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){


		if(!it->checkIfGradientAvailable()){     // evaluate constraint function value and read data

			it->evaluate(d);

		}
		else{

			it->evaluateAdjoint(d);

		}

		it->readEvaluateOutput(d);       // read constraint value


	}


}

void Optimizer::addConstraintValuesToDoEData(Design &d) const{

	unsigned int countConstraintWithGradient = 0;

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		std::string filenameCVS = it->getName()+".csv";

		if(ifDisplay){

			std::cout<<"Appending to data: "<<filenameCVS<<"\n";
		}

		if(it->checkIfGradientAvailable()){

			rowvec saveBuffer(2*dimension+1);

			copyRowVector(saveBuffer,d.designParameters);

			saveBuffer(dimension) = d.constraintTrueValues(it->getID());

			rowvec gradient = d.constraintGradients[countConstraintWithGradient];

			countConstraintWithGradient++;

			copyRowVector(saveBuffer,gradient,dimension+1);

			appendRowVectorToCSVData(saveBuffer,filenameCVS);   // read constraint data

		}else{

		  if(it->checkIfVectorConstraint()){

			  unsigned int ID = it->getVectorConstraintID();

			  long length = d.constraint_length[ID];

			  unsigned int dim = d.dimension;

			  rowvec saveBuffer(length+dim);

			  copyRowVector(saveBuffer,d.designParameters);

			  for(long i = dim; i< length+dim; i++){

				  saveBuffer(i) = d.constraint_vector[ID](i-dim);

			  }

			  appendRowVectorToCSVData(saveBuffer,filenameCVS);


          }else{

			 rowvec saveBuffer(dimension+1);

			 copyRowVector(saveBuffer,d.designParameters);

			 saveBuffer(dimension) = d.constraintTrueValues(it->getID());

			 appendRowVectorToCSVData(saveBuffer,filenameCVS);

		  }
	   }
	}
}


void Optimizer::estimateConstraints(CDesignExpectedImprovement &design) const{

	rowvec x = design.dv;

	assert(design.constraintValues.size() == numberOfConstraints);

	design.probability_con.zeros(numberOfConstraints);

	unsigned int constraintIt = 0;
	double mean = 0.0;
    double variance = 0.0;

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		it->interpolateWithVariance(x,&mean,&variance);

		design.constraintValues(constraintIt) = mean;

		if (variance != 0) {

			 design.probability_con(constraintIt)  = 1 - cdf(-mean/sqrt(variance),0.0,1.0);  // Compute the probability that the constraint function is satisfied.

		} else if(mean> 0)  {

			 design.probability_con(constraintIt) = 1;

        }else{
        	 design.probability_con(constraintIt) = 0;
		}

		constraintIt++;

		// cout << "design variable is " << x << endl;
	}
}


bool Optimizer::checkBoxConstraints(void) const{

	bool flagWithinBounds = true;

	for(unsigned int i=0; i<dimension; i++) {

		if(lowerBounds(i) >= upperBounds(i)) flagWithinBounds = false;
	}

	return flagWithinBounds;
}



bool Optimizer::checkConstraintFeasibility(rowvec constraintValues) const{

	bool flagFeasibility = true;
	unsigned int i=0;
	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		flagFeasibility = it->checkFeasibility(constraintValues(i));

		if(flagFeasibility == false) {

			break;
		}
		i++;
	}

	return flagFeasibility;
}



void Optimizer::print(void) const{

	std::cout<<"\nOptimizer Settings = \n\n";
	std::cout<<"Problem name : "<<name<<"\n";
	std::cout<<"Dimension    : "<<dimension<<"\n";
	std::cout<<"Type         : "<<optimizationType<<"\n";
	std::cout<<"Maximum number of function evaluations: " <<maxNumberOfSamples<<"\n";
	std::cout<<"Maximum number of iterations for EI maximization: " <<iterMaxEILoop<<"\n";


	objFun.print();

	printConstraints();

	if (constraintFunctions.empty()){

		std::cout << "Optimization problem does not have any constraints\n";
	}


}

void Optimizer::printConstraints(void) const{

	if(ifDisplay && !constraintFunctions.empty()){

		std::cout<< "List of constraints = \n";



		for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

			it->print();

		}

	}


}

void Optimizer::visualizeOptimizationHistory(void) const{

	if(dimension == 2){

		std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_2d_opthist.py "+ name;
#if 0
		cout<<python_command<<"\n";
#endif
		FILE* in = popen(python_command.c_str(), "r");

		fprintf(in, "\n");

	}


}


void Optimizer::initializeSurrogates(void){

	displayMessage("Initializing surrogate model for the objective function...\n");

	objFun.initializeSurrogate();

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		displayMessage("Initializing surrogate models for the constraint...\n");

		it->initializeSurrogate();

	}

	ifSurrogatesAreInitialized = true;

	displayMessage("Initialization is done...");

}


void Optimizer::trainSurrogates(void){

	//displayMessage("Training surrogate model for the objective function...\n");

	std::cout << "Training surrogate model for objective function..." << std::endl;

	objFun.trainSurrogate();

	if(ifDisplay) {

		objFun.printSurrogate();

	}


	if(constraintFunctions.size() !=0){

        std::cout << "Training surrogate model for the constraints..." << std::endl;
	}

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		it->trainSurrogate();

		if(ifDisplay) {

			it->printSurrogate();

		}

	}

	if(constraintFunctions.size() !=0){

		displayMessage("Model training for constraints is done...");

	}

}



void Optimizer::updateOptimizationHistory(Design d) {

	rowvec newSample(sampleDim+1);

	for(unsigned int i=0; i<dimension; i++) {

		newSample(i) = d.designParameters(i);

	}

	newSample(dimension) = d.trueValue;


	for(unsigned int i=0; i<numberOfConstraints; i++){

		newSample(i+dimension+1) = 	d.constraintTrueValues(i);
	}
	newSample(sampleDim) = d.improvementValue;



	optimizationHistory.insert_rows( optimizationHistory.n_rows, newSample );
	appendRowVectorToCSVData(newSample,"optimizationHistory.csv");

#if 0
	printf("optimizationHistory:\n");
	optimizationHistory.print();

#endif


}


void Optimizer::addPenaltyToExpectedImprovementForConstraints(CDesignExpectedImprovement &designCalculated) const{

	if(numberOfConstraints > 0){

		estimateConstraints(designCalculated);

		for (unsigned int i=0; i< numberOfConstraints ; i++){    // Created by Kai

			designCalculated.valueExpectedImprovement = designCalculated.valueExpectedImprovement*designCalculated.probability_con(i);   // Compute the constrained (g(x) > 0) Expected improvement function

		}

		// bool ifConstraintsSatisfied = checkConstraintFeasibility(designCalculated.constraintValues);              // Modified by Kai

		//  if(!ifConstraintsSatisfied){

		// 	designCalculated.valueExpectedImprovement = 0.0;

		// }

	}


}


bool Optimizer::ifConstrained(void) const{

	if(numberOfConstraints > 0) return true;
	else return false;

}


void Optimizer::computeConstraintsandPenaltyTerm(Design &d) {


	if(ifConstrained()){

		displayMessage("Evaluating constraints...\n");

		evaluateConstraints(d);

		double penaltyTerm = 0.0;

		bool ifConstraintsSatisfied = checkConstraintFeasibility(d.constraintTrueValues);

		if(!ifConstraintsSatisfied){


			if(ifDisplay){

				std::cout<<"The new sample does not satisfy all the constraints\n";

			}



			if(optimizationType == "minimize"){

				penaltyTerm = LARGE;
				d.isDesignFeasible = false;
			}
			else if(optimizationType == "maximize"){

				penaltyTerm = -LARGE;
				d.isDesignFeasible = false;
			}

		}

		d.objectiveFunctionValue = d.trueValue ;  // Modified by kai
		// d.objectiveFunctionValue = d.trueValue + penaltyTerm;

	}

	displayMessage("Evaluating constraints is done...\n");

}

void Optimizer::addConstraintValuesToData(Design &d){

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		it->addDesignToData(d);

	}


}

void Optimizer::checkIfSettingsAreOK(void) const{

	if (maxNumberOfSamples == 0){

		fprintf(stderr, "ERROR: Maximum number of samples is not set for the optimization!\n");
		cout<<"maxNumberOfSamples = "<<maxNumberOfSamples<<"\n";
		abort();
	}



	if(checkBoxConstraints() == false){

		fprintf(stderr, "ERROR: Box constraints are not set properly!\n");
		abort();

	}
}


void Optimizer::zoomInDesignSpace(void){

	displayMessage("Zooming in design space...\n");

	findTheGlobalOptimalDesign();

#if 0
	globalOptimalDesign.print();
#endif
	vec dx(dimension);

	for(unsigned int i=0; i<dimension; i++){

		dx(i) =  upperBoundsForEIMaximization(i) - lowerBoundsForEIMaximization(i);

	}

#if 0
	dx.print();
#endif

	rowvec dvNormalized = normalizeRowVector(globalOptimalDesign.designParameters, lowerBounds, upperBounds);

#if 0
	printVector(dvNormalized,"dvNormalized");
#endif

	for(unsigned int i=0; i<dimension; i++){

		double delta = dx(i)*zoomInFactor;

		if(delta < 10E-5){

			delta = 10E-5;
		}

		lowerBoundsForEIMaximization(i) =  dvNormalized(i) - delta;
		upperBoundsForEIMaximization(i) =  dvNormalized(i) + delta;

		if(lowerBoundsForEIMaximization(i) < 0.0) {

			lowerBoundsForEIMaximization(i) = 0.0;

		}

		if(upperBoundsForEIMaximization(i) > 1.0/dimension) {

			upperBoundsForEIMaximization(i) = 1.0/dimension;

		}

	}
#if 0
	printVector(lowerBoundsForEIMaximization,"lowerBoundsForEIMaximization" );
	printVector(upperBoundsForEIMaximization,"upperBoundsForEIMaximization");
#endif

	zoomInFactor = zoomInFactor*0.75;
}



void Optimizer::findTheGlobalOptimalDesign(void){

	displayMessage("Finding the global design...\n");

	/* we take the sample, which has the max improvement value */

	uword indexMin = index_max(optimizationHistory.col(sampleDim));

	rowvec bestSample = optimizationHistory.row(indexMin);

	rowvec dv(dimension);

	for(unsigned int i=0; i<dimension; i++){

		dv(i) = bestSample(i);
	}

	globalOptimalDesign.ID = indexMin;
	globalOptimalDesign.tag = "Global optimum design";
	globalOptimalDesign.designParameters  = dv;
	globalOptimalDesign.trueValue = bestSample(dimension);
	globalOptimalDesign.objectiveFunctionValue = bestSample(dimension);
	globalOptimalDesign.improvementValue = bestSample(sampleDim);

	rowvec constraintValues(numberOfConstraints);
	for(unsigned int i=0; i<numberOfConstraints; i++){

		constraintValues(i) = bestSample(i+dimension+1);
	}

	globalOptimalDesign.constraintTrueValues = constraintValues;
	displayMessage("Finding the global design is done...\n");

}

/* These designs (there can be more than one) are found by maximizing the expected
 *  Improvement function and taking the constraints into account
 */

void Optimizer::findTheMostPromisingDesign(unsigned int howManyDesigns){

	 assert(ifSurrogatesAreInitialized);

	 theMostPromisingDesigns.clear();

	 CDesignExpectedImprovement designWithMaxEI(dimension,numberOfConstraints);

     rowvec initial_dv;

     initial_dv.zeros(dimension);

     initial_dv = (lowerBounds.t()+upperBounds.t())/2;   // initial value of design parameter

     designWithMaxEI.dv = initial_dv;

 #pragma omp parallel for

	for(unsigned int iterEI = 0; iterEI <iterMaxEILoop; iterEI++ ){   // global search

		CDesignExpectedImprovement designToBeTried(dimension,numberOfConstraints);

		designToBeTried.generateRandomDesignVector(lowerBoundsForEIMaximization, upperBoundsForEIMaximization);

		objFun.calculateExpectedImprovement(designToBeTried);

		addPenaltyToExpectedImprovementForConstraints(designToBeTried);

		// cout << " Expected improvement value is " << designToBeTried.valueExpectedImprovement<< endl;

#if 0
		designToBeTried.print();
#endif
		if(designToBeTried.valueExpectedImprovement > designWithMaxEI.valueExpectedImprovement){

			designWithMaxEI = designToBeTried;
#if 0
			printf("A design with a better EI value has been found\n");
			designToBeTried.print();
#endif
		}

	}

	#pragma omp parallel for

	for(unsigned int iterEI = 0; iterEI <iterMaxEILoop; iterEI++ ){             // local search

		CDesignExpectedImprovement designToBeTried(dimension,numberOfConstraints);

		designToBeTried.generateRandomDesignVectorAroundASample(designWithMaxEI.dv, lowerBoundsForEIMaximization, upperBoundsForEIMaximization);

		objFun.calculateExpectedImprovement(designToBeTried);

		addPenaltyToExpectedImprovementForConstraints(designToBeTried);          // Modified by Kai

		// cout << " Expected improvement value is " << designToBeTried.valueExpectedImprovement<< endl;

#if 0
		designToBeTried.print();
#endif
		if(designToBeTried.valueExpectedImprovement > designWithMaxEI.valueExpectedImprovement){

			designWithMaxEI = designToBeTried;
#if 0
			printf("A design with a better EI value has been found (second loop) \n");
			designToBeTried.print();
#endif
		}

	}


	theMostPromisingDesigns.push_back(designWithMaxEI);

}


CDesignExpectedImprovement Optimizer::getDesignWithMaxExpectedImprovement(void) const{

	return this->theMostPromisingDesigns.front();


}

/* calculate the gradient of the Expected Improvement function
 * w.r.t design variables by finite difference approximations */
rowvec Optimizer::calculateEIGradient(CDesignExpectedImprovement &currentDesign) const{


	rowvec gradient(dimension);


	for(unsigned int i=0; i<dimension; i++){
#if 0
		printf("dv:\n");
		dvGradientSearch.print();
#endif

		double dvSave = currentDesign.dv(i);
#if 0
		printf("epsilon_EI = %15.10f\n",epsilon_EI);
#endif

		double epsilon = currentDesign.dv(i)*0.00001;
		currentDesign.dv(i) += epsilon;

#if 0
		printf("dv perturbed:\n");
		dvPerturbed.print();
#endif

		objFun.calculateExpectedImprovement(currentDesign);

		addPenaltyToExpectedImprovementForConstraints(currentDesign); // Created by Kai

		double EIplus = currentDesign.valueExpectedImprovement;

		currentDesign.dv(i) -= 2*epsilon;

		objFun.calculateExpectedImprovement(currentDesign);

		addPenaltyToExpectedImprovementForConstraints(currentDesign); // Created by Kai

		double EIminus = currentDesign.valueExpectedImprovement;;


		/* obtain the forward finite difference quotient */
		double fdVal = (EIplus - EIminus)/(2*epsilon);
		gradient(i) = fdVal;
		currentDesign.dv(i) = dvSave;


	} /* end of finite difference loop */
#if 0
	printf("Gradient vector:\n");
	gradEI.print();
#endif

	return gradient;
}



CDesignExpectedImprovement Optimizer::MaximizeEIGradientBased(CDesignExpectedImprovement initialDesign) const {

	rowvec gradEI(dimension);
	double stepSize0 = 0.001;
	double stepSize = 0.0;

	objFun.calculateExpectedImprovement(initialDesign);

	addPenaltyToExpectedImprovementForConstraints(initialDesign);

	double EI0 = initialDesign.valueExpectedImprovement;

	CDesignExpectedImprovement bestDesign = initialDesign;

	bool breakOptimization = false;

	for(unsigned int iterGradientSearch=0; iterGradientSearch<iterGradientEILoop; iterGradientSearch++){

#if 0
		printf("\nGradient search iteration = %d\n", iterGradientSearch);
#endif

		gradEI = calculateEIGradient(bestDesign);

		/* save the design vector */
		CDesignExpectedImprovement dvLineSearchSave = bestDesign ;

#if 0
		printf("Line search...\n");
#endif

		stepSize = stepSize0;

		while(1){

			/* design update */

			bestDesign.gradientUpdateDesignVector(gradEI,stepSize);

			objFun.calculateExpectedImprovement(bestDesign);

			addPenaltyToExpectedImprovementForConstraints(bestDesign);


#if 0
			printf("EI_LS = %15.10f\n",bestDesign.valueExpectedImprovement );

#endif

			/* if ascent is achieved */
			if(bestDesign.valueExpectedImprovement > EI0){
#if 0
				printf("Ascent is achieved with difference = %15.10f\n", bestDesign.valueExpectedImprovement -  EI0);

				bestDesign.print();
#endif
				EI0 = bestDesign.valueExpectedImprovement;
				break;
			}

			else{ /* else halve the stepsize and set design to initial */

				stepSize = stepSize * 0.5;
				bestDesign = dvLineSearchSave;
#if 0
				printf("stepsize = %15.10f\n",stepSize);

#endif
				if(stepSize < 10E-12) {
#if 0
					printf("The stepsize is getting too small!\n");
#endif

					breakOptimization = true;
					break;
				}
			}

		}

		if(breakOptimization) break;

	} /* end of gradient-search loop */


	return bestDesign;


}
void Optimizer::prepareOptimizationHistoryFile(void) const{

	std::string header;
	for(unsigned int i=0; i<dimension; i++){

		header+="x";
		header+=std::to_string(i+1);
		header+=",";

	}

	header+="Objective Function,";


	for(unsigned int i=0; i<this->numberOfConstraints; i++){
		header+="Constraint";
		header+=std::to_string(i+1);

		header+=",";

	}

	header+="Improvement";
	header+="\n";

	std::ofstream optimizationHistoryFile;
	optimizationHistoryFile.open (optimizationHistoryFileName);
	optimizationHistoryFile << header;
	optimizationHistoryFile.close();

}

void Optimizer::clearOptimizationHistoryFile(void) const{

	remove(optimizationHistoryFileName.c_str());

}


void Optimizer::EfficientGlobalOptimization(void){

	checkIfSettingsAreOK();

	if(!isHistoryFileInitialized){
		clearOptimizationHistoryFile();
		prepareOptimizationHistoryFile();
	}

	/* main loop for optimization */

	unsigned int simulationCount = 0;
	unsigned int iterOpt=0;

	initializeSurrogates();

	howOftenTrainModels = 1 ;

	clock_t start, finish;

	while(1){

		iterOpt++;

        cout <<"Number of iterations is " << iterOpt <<endl;

#if 0
		printf("Optimization Iteration = %d\n",iterOpt);
#endif


		if(simulationCount%howOftenTrainModels == 0) {

			  trainSurrogates();

		}

		/*if(iterOpt%10 == 0){

			zoomInDesignSpace();

		}*/

		 start = clock();

		    findTheMostPromisingDesign();    // find the optimal next point with maximal EI

	        finish = clock();

		 cout << "The time for finding most promising design point is " << (double)(finish-start)/CLOCKS_PER_SEC  <<endl;


		// CDesignExpectedImprovement optimizedDesignGradientBased = MaximizeEIGradientBased(theMostPromisingDesigns.at(0));  // Need further revision

	      CDesignExpectedImprovement optimizedDesignGradientBased = theMostPromisingDesigns.at(0);


#if 0
		optimizedDesignGradientBased.print();
#endif

		rowvec best_dvNorm = optimizedDesignGradientBased.dv;

		rowvec best_dv = normalizeRowVectorBack(best_dvNorm, lowerBounds, upperBounds);

		double estimatedBestdv = objFun.interpolate(best_dvNorm);


#if 0
		printf("The most promising design (not normalized):\n");
		best_dv.print();
		std::cout<<"Estimated objective function value = "<<estimatedBestdv<<"\n";

#endif

		Design currentBestDesign(best_dv);

		currentBestDesign.setNumberOfConstraints(numberOfConstraints);
		currentBestDesign.saveDesignVector(designVectorFileName);

		/* now make a simulation for the most promising design */

		if(!objFun.checkIfGradientAvailable()) {

			objFun.evaluate(currentBestDesign);
		}
		else{

			objFun.evaluateAdjoint(currentBestDesign);
		}

		objFun.readEvaluateOutput(currentBestDesign);

		objFun.addDesignToData(currentBestDesign);

		computeConstraintsandPenaltyTerm(currentBestDesign);  // Currently, we do not penalize the constraint.

		calculateImprovementValue(currentBestDesign);

		if(currentBestDesign.checkIfHasNan()){

			cout<<"ERROR: NaN while reading external executable outputs!\n";
			abort();

		}
#if 0
		currentBestDesign.print();
#endif

		addConstraintValuesToData(currentBestDesign);

		updateOptimizationHistory(currentBestDesign);

		findTheGlobalOptimalDesign();

		globalOptimalDesign.saveToAFile(globalOptimumDesignFileName);


		if(ifDisplay){

			std::cout<<"##########################################\n";
			std::cout<<"Optimization Iteration = "<<iterOpt<<"\n";
			currentBestDesign.print();
			std::cout<<"\n\n";

		}

		simulationCount ++;

		/* terminate optimization */

		if(simulationCount >= maxNumberOfSamples){


			if(ifDisplay){

				printf("number of simulations > max_number_of_samples! Optimization is terminating...\n");

				std::cout<<"##########################################\n";
				std::cout<<"Global best design = \n";
				globalOptimalDesign.print();
				std::cout<<"\n\n";

			}

			if(ifVisualize){

				visualizeOptimizationHistory();
			}

			break;
		}

	} /* end of the optimization loop */
}

void Optimizer::cleanDoEFiles(void) const{

	std::string fileNameObjectiveFunction = objFun.getName()+".csv";


	if(file_exist(fileNameObjectiveFunction)){

		remove(fileNameObjectiveFunction.c_str());
	}

	for (auto it = constraintFunctions.begin(); it != constraintFunctions.end(); it++){

		std::string fileNameConstraint = it->getName()+".csv";
		if(file_exist(fileNameConstraint)){

			remove(fileNameConstraint.c_str());
		}


	}


}

void Optimizer::calculateImprovementValue(Design &d){


	if(d.isDesignFeasible){

		if(!IfinitialValueForObjFunIsSet){

			initialobjectiveFunctionValue  = d.objectiveFunctionValue;
			IfinitialValueForObjFunIsSet = true;

		}


		if(optimizationType == "minimize"){

			if(d.objectiveFunctionValue < initialobjectiveFunctionValue){

				d.improvementValue = initialobjectiveFunctionValue - d.objectiveFunctionValue;

			}

		}
		if(optimizationType == "maximize"){

			if(d.objectiveFunctionValue > initialobjectiveFunctionValue){

				d.improvementValue = d.objectiveFunctionValue - initialobjectiveFunctionValue;

			}

		}


	}


}



void Optimizer::performDoE(unsigned int howManySamples, DoE_METHOD methodID){

	if(ifDisplay){

		std::cout<<"performing DoE...\n";
	}



	if(!ifBoxConstraintsSet){

		cout<<"ERROR: Cannot run DoE before the box-constraints are set!\n";
		abort();
	}



	if(!isHistoryFileInitialized){

		clearOptimizationHistoryFile();
		prepareOptimizationHistoryFile();
		isHistoryFileInitialized = true;

	}

	mat sampleCoordinates;

	if(methodID == LHS){

		LHSSamples DoE(dimension, lowerBounds, upperBounds, howManySamples);

		std::string filename= this->name + "_samples.csv";
		DoE.saveSamplesToCSVFile(filename);                     // export input samples
		sampleCoordinates = DoE.getSamples();
	}
	else{

		cout<<"ERROR: Cannot run DoE with any option other than LHS!\n";
		abort();

	}


#if 0
	printMatrix(sampleCoordinates,"sampleCoordinates");
#endif


	for(unsigned int sampleID=0; sampleID<howManySamples; sampleID++){

		if(ifDisplay){

			std::cout<<"\n##########################################\n";
			std::cout<<"Evaluating sample "<<sampleID<<"\n";

		}

		rowvec dv = sampleCoordinates.row(sampleID);
		Design currentDesign(dv);

		currentDesign.setNumberOfConstraints(numberOfConstraints);   // constraint function number
		currentDesign.saveDesignVector(designVectorFileName);

		std::string filenameCVS = objFun.getName()+".csv";

		if(ifDisplay){

			std::cout<<"Appending to data: "<<filenameCVS<<"\n";

		}

		if(!objFun.checkIfGradientAvailable()) {

			objFun.evaluate(currentDesign);               // objective function

			objFun.readEvaluateOutput(currentDesign);

			rowvec temp(dimension+1);
			copyRowVector(temp,dv);
			temp(dimension) = currentDesign.trueValue;
			appendRowVectorToCSVData(temp,filenameCVS);


		}
		else{

			objFun.evaluateAdjoint(currentDesign);        // objective function and its gradient
			objFun.readEvaluateOutput(currentDesign);
			rowvec temp(2*dimension+1);
			copyRowVector(temp,currentDesign.designParameters);
			temp(dimension) = currentDesign.trueValue;
			copyRowVector(temp,currentDesign.gradient, dimension+1);
			appendRowVectorToCSVData(temp,filenameCVS);


		}

		computeConstraintsandPenaltyTerm(currentDesign);   // constraint function

		calculateImprovementValue(currentDesign);

		addConstraintValuesToDoEData(currentDesign);

		updateOptimizationHistory(currentDesign);


		if(ifDisplay){

			currentDesign.print();

		}

	} /* end of sample loop */

}

void Optimizer::displayMessage(std::string inputString) const{


	if(ifDisplay){

		std::cout<<inputString<<"\n";


	}


}



