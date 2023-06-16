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

#include <stdio.h>
#include <math.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>
#include "auxiliary_functions.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "test_functions.hpp"
#include "optimization.hpp"
#include "objective_function.hpp"
#include "lhs.hpp"
#include "bounds.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


using namespace arma;


ObjectiveFunctionDefinition::ObjectiveFunctionDefinition(void){


}


ObjectiveFunctionDefinition::ObjectiveFunctionDefinition(std::string name){

	this->name = name;

}


void ObjectiveFunctionDefinition::print(void) const{

	std::cout<<"\nObjective function definition = \n";
	std::cout<< "Name = "<<name<<"\n";
	std::cout<< "Design vector filename = "<<designVectorFilename<<"\n";


	if(ifMultiLevel == false){


		std::cout<< "Output filename = "<<outputValueFilename<<"\n";
		std::cout<< "Executable name = "<<executableName<<"\n";

		if(isNotEmpty(path)){

			std::cout<< "Executable path = "<<path<<"\n";

		}

		if(!marker.empty()){

			std::cout<< "Marker = "<<marker<<"\n";


		}

	}

	else{

		std::cout<<"Multilevel option is active...\n";


	}




}



ObjectiveFunction::ObjectiveFunction(std::string objectiveFunName, unsigned int dimension)
: surrogateModel(objectiveFunName),surrogateModelGradient(objectiveFunName),surrogateModelML(objectiveFunName),gekModel(objectiveFunName),sgekModel(objectiveFunName){


	dim = dimension;
	name = objectiveFunName;

	fileNameTrainingDataForSurrogate = name + ".csv";


}


ObjectiveFunction::ObjectiveFunction(){


}

/* void ObjectiveFunction::bindSurrogateModel(SURROGATE_MODEL modelType){

	assert(ifDefinitionIsSet);

	surrogateModelType = modelType;

	switch(modelType) {

		  case GRADIENT_ENHANCED_KRIGING:

			  gekModel.setNameOfInputFile(fileNameTrainingDataForSurrogate);
			  output.printMessage("Binding the surrogate model with the gradient enhanced Kriging modeĺ...");
			  surrogate = &gekModel;
			  surrogate->setGradientsOn();
		 	  break;

		  case SLICED_GRADIENT_ENHANCED_KRIGING:

			  gekModel.setNameOfInputFile(fileNameTrainingDataForSurrogate);
			  output.printMessage("Binding the surrogate model with the sliced gradient enhanced Kriging modeĺ...");
			  surrogate = &sgekModel;
			  surrogate->setGradientsOn();
		 	  break;

		  case ORDINARY_KRIGING:

			  surrogateModel.setNameOfInputFile(fileNameTrainingDataForSurrogate);
			  output.printMessage("Binding the surrogate model with the ordinary Kriging modeĺ...");
			  surrogate = &surrogateModel;
			  break;

		  case UNIVERSAL_KRIGING:
			  surrogateModel.setLinearRegressionOn();
			  surrogateModel.setNameOfInputFile(fileNameTrainingDataForSurrogate);
			  output.printMessage("Binding the surrogate model with the universal Kriging modeĺ...");
			  surrogate = &surrogateModel;

		    break;

		  case AGGREGATION:
			  surrogateModel.setNameOfInputFile(fileNameTrainingDataForSurrogate);
			  output.printMessage("Binding the surrogate model with the Agrregation modeĺ...");
			  surrogate = &surrogateModelGradient;
			  break;

		  case MULTI_LEVEL:

			  output.printMessage("Binding the surrogate model with the Multi-Level modeĺ...");
			  surrogateModelML.setinputFileNameHighFidelityData(fileNameInputRead);
			  surrogateModelML.setinputFileNameLowFidelityData(fileNameInputReadLowFi);
			  surrogate = &surrogateModelML;
			  break;

		  default:

			  outputToScreen.printErrorMessageAndAbort("Unknown modelType for the surrogate model!");

		}
}*/


 void ObjectiveFunction::bindSurrogateModel(void){

	assert(ifDefinitionIsSet);

	if(ifMultilevel){
		cout << "Fitting function with Multi-Level modeĺ" << endl;
		output.printMessage("Binding the surrogate model with the Multi-Level modeĺ...");
		surrogateModelML.setinputFileNameHighFidelityData(fileNameOutputValueRead);
		surrogateModelML.setinputFileNameLowFidelityData(fileNameInputReadLowFi);
		surrogate = &surrogateModelML;

	}

	else if(ifGradientAvailable == false){
		cout << "Fitting function with Kriging model" << endl;
		surrogateModel.setNameOfInputFile(fileNameTrainingDataForSurrogate);
		output.printMessage("Binding the surrogate model with the Kriging modeĺ...");
		surrogate = &surrogateModel;

		if (ifVectorOutput == true){
			surrogate->setVectorConstraintOn();
			surrogate->setConstraintLength(constraint_length);
		}

	}

	else if (surrogatetype == "gradient_enhanced_kriging"){
		cout << "Fitting function with gradient enhanced kriging model" << endl;
		gekModel.setNameOfInputFile(fileNameTrainingDataForSurrogate);
		output.printMessage("Binding the surrogate model with the gradient enhanced Kriging modeĺ...");
	    surrogate = &gekModel;
		surrogate->setGradientsOn();

	    }

	else if (surrogatetype == "sliced_gradient_enhanced_kriging"){
		cout << "Fitting function with sliced gradient enhanced kriging model" << endl;
		gekModel.setNameOfInputFile(fileNameTrainingDataForSurrogate);
		output.printMessage("Binding the surrogate model with the sliced gradient enhanced Kriging modeĺ...");
		surrogate = &sgekModel;
		surrogate->setGradientsOn();

		 }

	else if (surrogatetype == "agrregation_model"){
		cout << "Fitting function with Agrregation model" << endl;
		surrogateModel.setNameOfInputFile(fileNameTrainingDataForSurrogate);
		surrogateModel.setNameOfInputFile(fileNameTrainingDataForSurrogate);
		output.printMessage("Binding the surrogate model with the Agrregation modeĺ...");
		surrogate = &surrogateModelGradient;

	   }

	else {
			cout << "Error ! Surrogate model type has not been set !" << endl;

		   }
}


void ObjectiveFunction::setParametersByDefinition(ObjectiveFunctionDefinition definition){

	executableName = definition.executableName;
	executablePath = definition.path;
	jsonFile = definition.jsonFile;
	fileNameDesignVector = definition.designVectorFilename;
	fileNameOutputValueRead = definition.outputValueFilename;
	fileNameOutputGradRead = definition.outputGradFilename;
	ifGradientAvailable = definition.ifGradient;
	surrogatetype = definition.surrogatetype;

	if(isNotEmpty(definition.marker)){

		readMarker = definition.marker;
		ifMarkerIsSet = true;

	}


	if(isNotEmpty(definition.markerForGradient)){


		readMarkerAdjoint = definition.markerForGradient;
		ifAdjointMarkerIsSet = true;

	}


	if(definition.ifMultiLevel){

		ifMultilevel = definition.ifMultiLevel;
		executableNameLowFi = definition.executableNameLowFi;
		fileNameInputReadLowFi = definition.outputFilenameLowFi;
		executablePathLowFi = definition.pathLowFi;
		readMarkerLowFi = definition.markerLowFi;
		readMarkerAdjointLowFi = definition.markerForGradientLowFi;


	}


	ifDefinitionIsSet = true;



}

void ObjectiveFunction::setFunctionPointer(double (*objFun)(double *)){

	objectiveFunPtr = objFun;
	ifFunctionPointerIsSet = true;

}

void ObjectiveFunction::setFunctionPointer(double (*objFun)(double *, double *)){

	objectiveFunAdjPtr = objFun;
	ifFunctionPointerIsSet = true;

}




void ObjectiveFunction::setGradientOn(void){

	ifGradientAvailable = true;

}
void ObjectiveFunction::setGradientOff(void){

	ifGradientAvailable = false;

}

void ObjectiveFunction::setVectorOutputOn(void){   // Kai

	ifVectorOutput = true;

}

void ObjectiveFunction::setVectorOutputOff(void){  // Kai

	ifVectorOutput = false;

}

void ObjectiveFunction::setConstraintLength(int length){
	constraint_length = length;
}

mat ObjectiveFunction::getPodBasis(void) const{
	return pod_basis;
}

void ObjectiveFunction::setDisplayOn(void){

	output.ifScreenDisplay = true;
}
void ObjectiveFunction::setDisplayOff(void){

	output.ifScreenDisplay = false;

}



void ObjectiveFunction::setNumberOfTrainingIterationsForSurrogateModel(unsigned int nIter){

	numberOfIterationsForSurrogateTraining = nIter;

}

void ObjectiveFunction::setFileNameReadOutputValue(std::string fileName){

	assert(!fileName.empty());
	fileNameOutputValueRead = fileName;

}

void ObjectiveFunction::setFileNameReadOutputGrad(std::string fileName){

	assert(!fileName.empty());
	fileNameOutputGradRead = fileName;

}

void ObjectiveFunction::setFileNameDesignVector(std::string fileName){

	assert(!fileName.empty());
	fileNameDesignVector = fileName;

}

void ObjectiveFunction::setExecutablePath(std::string path){

	assert(!path.empty());
	executablePath = path;

}

void ObjectiveFunction::setExecutableName(std::string exeName){

	assert(!exeName.empty());
	executableName = exeName;

}

void ObjectiveFunction::setParameterBounds(vec lb, vec ub){

	assert(dim == lb.size());
	assert(dim == ub.size());
	lowerBounds = lb;
	upperBounds = ub;

	ifParameterBoundsAreSet = true;
}



KrigingModel ObjectiveFunction::getSurrogateModel(void) const{

	return surrogateModel;

}

AggregationModel ObjectiveFunction::getSurrogateModelGradient(void) const{

	return surrogateModelGradient;

}

GEKModel ObjectiveFunction::getGEKModel(void) const{

	return gekModel;

}

SGEKModel ObjectiveFunction::getSGEKModel(void) const{

	return sgekModel;

}

MultiLevelModel ObjectiveFunction::getSurrogateModelML(void) const{

	return surrogateModelML;

}


void ObjectiveFunction::setReadMarker(std::string marker){

	assert(isNotEmpty(marker));

	readMarker = marker;
	ifMarkerIsSet = true;

}
std::string ObjectiveFunction::getReadMarker(void) const{

	return readMarker;
}
void ObjectiveFunction::setReadMarkerAdjoint(std::string marker){

	assert(isNotEmpty(marker));

	readMarkerAdjoint = marker;
	ifAdjointMarkerIsSet = true;

}

std::string ObjectiveFunction::getReadMarkerAdjoint(void) const{

	return readMarkerAdjoint;
}

void ObjectiveFunction::initializeSurrogate(void){

	assert(ifParameterBoundsAreSet);
	assert(ifDefinitionIsSet);

	bindSurrogateModel();

	Bounds boxConstraints(lowerBounds,upperBounds);

	surrogate->setBoxConstraints(boxConstraints);

	surrogate->readData();

	surrogate->normalizeData();

	pod_basis = surrogate->getPodBasis();

	rank = surrogate->readRank();

	surrogate->initializeSurrogateModel();

	surrogate->setNumberOfTrainingIterations(numberOfIterationsForSurrogateTraining);

#if 0
	surrogate->printSurrogateModel();
#endif


	ifInitialized = true;

}

void ObjectiveFunction::trainSurrogate(void){

	assert(ifInitialized);

	surrogate->train();

}



void ObjectiveFunction::saveDoEData(std::vector<rowvec> data) const{

	std::string fileName = surrogate->getNameOfInputFile();

	std::ofstream myFile(fileName);

	myFile.precision(10);

	for(unsigned int i = 0; i < data.size(); ++i)
	{
		rowvec v=  data.at(i);


		for(unsigned int j= 0; j<v.size(); j++){

			myFile <<v(j)<<",";
		}

		myFile << "\n";
	}


	myFile.close();


}


void ObjectiveFunction::calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const{


	surrogate->calculateExpectedImprovement(designCalculated);

}


bool ObjectiveFunction::checkIfGradientAvailable(void) const{

	return ifGradientAvailable;

}

bool ObjectiveFunction::checkIfVectorConstraint(void) const{

	return ifVectorOutput;

}


std::string ObjectiveFunction::getExecutionCommand(void) const{

	assert(isNotEmpty(executableName));

	std::string runCommand;

    if (executableName == "FdmSolver"){

    	runCommand = executablePath +"/" + executableName + " -f " + jsonFile;   // For induheat

    }

	else if(isNotEmpty(executablePath)) {

		runCommand = executablePath +"/" + executableName;
	}
	else{

		runCommand = "./" + executableName;
	}

	return runCommand;


}

std::string ObjectiveFunction::getExecutionCommandLowFi(void) const{

	assert(ifMultilevel);
	assert(isNotEmpty(executableNameLowFi));

	std::string runCommand;

	if(isNotEmpty(executablePathLowFi)) {

		runCommand = executablePathLowFi +"/" + executableNameLowFi;
	}
	else{

		runCommand = "./" + executableNameLowFi;
	}

	return runCommand;

}





void ObjectiveFunction::addDesignToData(Design &d){

	rowvec newsample;

	if(ifGradientAvailable){

		newsample = d.constructSampleObjectiveFunctionWithGradient();


	}
	else{

		newsample = d.constructSampleObjectiveFunction();

	}

   cout << "The optimal design point found is " << newsample <<  endl;

	surrogate->addNewSampleToData(newsample);

}


void ObjectiveFunction::readOutputWithoutMarkers(Design &outputDesignBuffer) const{

	std::ifstream inputFileStream(fileNameOutputValueRead, ios::in);

	if (!inputFileStream.is_open()) {

		cout << "ERROR: There was a problem opening the input file!\n";
		abort();
	}

	double functionValue;
	inputFileStream >> functionValue;
	outputDesignBuffer.trueValue = functionValue;
	outputDesignBuffer.objectiveFunctionValue = functionValue;
	inputFileStream.close();

	std::string line;
	std::vector<double> obj_grad;

	if(ifGradientAvailable){

		assert(!this->fileNameOutputGradRead.empty());

	    std::ifstream inputFileStream1(fileNameOutputGradRead, ios::in);

	    if (!inputFileStream1.is_open()) {
				cout << "ERROR: There was a problem opening the input file!\n";
			    abort();
		 }

	       while(std::getline(inputFileStream1, line)){
	              double id = static_cast<double>(std::stod(line) );  // line by line
	              obj_grad.push_back(id);
	        }
		   inputFileStream1.close();

		   for (unsigned int i=0; i< dim; i++){
		       outputDesignBuffer.gradient(i) =obj_grad[i];
		   }
	}


}

void ObjectiveFunction::readEvaluateOutput(Design &d){

	assert(!this->fileNameOutputValueRead.empty());

	assert(d.dimension == dim);


	if(ifMarkerIsSet && ifGradientAvailable){

		if(!ifAdjointMarkerIsSet){

			cout << "ERROR: Adjoint marker is not set for: "<< this->name<<"\n";
			abort();

		}

	}

	if(ifAdjointMarkerIsSet == true && ifGradientAvailable == false){


		cout << "ERROR: Adjoint marker is set for the objective function but gradient is not available!\n";
		cout << "Did you set GRADIENT_AVAILABLE properly?\n";
		abort();
	}


	readOutputWithoutMarkers(d);

	/* if(ifMarkerIsSet == false){

		// If there is not any marker, just reads the functional value (and gradient) from the input file

		readOutputWithoutMarkers(d);
	}

	else{

		for( std::string line; getline( ifile, line ); ){  //

			size_t found = line.find(readMarker+" ");

			if (found != std::string::npos){

				line = removeSpacesFromString(line);
				line.erase(0,found+1+this->readMarker.size());  // erase from 0 to data location

				d.trueValue = stod(line);                       //  stod : transform string to double
				d.objectiveFunctionValue = stod(line);

			}

			if(this->ifGradientAvailable){


				size_t found2 = line.find(readMarkerAdjoint+" ");



				if (found2!=std::string::npos){
					line = removeSpacesFromString(line);
					line.erase(0,found2+1+readMarkerAdjoint.size());
					vec values = getDoubleValuesFromString(line,',');
					assert(values.size() == dim);


					for(unsigned int i=0; i<dim; i++){

						d.gradient(i) = values(i);
					}


				}


			}
		}

	}

	ifile.close(); */



}

void ObjectiveFunction::evaluate(Design &d){

	if(ifFunctionPointerIsSet){

		rowvec x= d.designParameters;
		double functionValue =  objectiveFunPtr(x.memptr());
		d.trueValue = functionValue;
		d.objectiveFunctionValue = functionValue;

	}

	else if (executableName != "None" && fileNameDesignVector != "None"){


		std::string runCommand = getExecutionCommand();

#if 0
		cout<<"calling a system command\n";
#endif
		system(runCommand.c_str());

	}
	else{

		cout<<"ERROR: Cannot evaluate the objective function. Check settings!\n";
		abort();
	}


}


void ObjectiveFunction::evaluateAdjoint(Design &d){


	assert(ifGradientAvailable);

	if( ifFunctionPointerIsSet){
		rowvec x= d.designParameters;
		rowvec xb(dim);
		xb.fill(0.0);

		double functionValue =  objectiveFunAdjPtr(x.memptr(),xb.memptr());

		d.trueValue = functionValue;
		d.objectiveFunctionValue = functionValue;
		d.gradient = xb;

	}

	else if (executableName != "None" && fileNameDesignVector != "None"){

		std::string runCommand = getExecutionCommand();

		system(runCommand.c_str());

	}
	else{

		cout<<"ERROR: Cannot evaluate the objective function. Check settings!\n";
		abort();
	}
}


void ObjectiveFunction::interpolateWithVariance(rowvec x, double *mean, double *variance) const{


	  if (ifVectorOutput){

		  vec mean_vec;
		  vec variance_vec;

		  vec mean_vec1(constraint_length);
		  vec variance_vec1(constraint_length);

		  mean_vec1.zeros();
		  variance_vec1.zeros();

		  surrogate->interpolateWithVariance_vec(x, mean_vec, variance_vec);

		  mat basis = getPodBasis();

		  rowvec meanvector = surrogate->readOutputMeanVector();
		  rowvec stdvector  = surrogate->readOutputStdVector();

		  for (unsigned int i = 0; i< rank; i++ ){       // recover the full state solution

			  if (stdvector(i) != 0){

			    mean_vec(i) = mean_vec(i)*stdvector(i) + meanvector(i);

			    variance_vec(i) = variance_vec(i)*stdvector(i)*stdvector(i);

			  }
			  else {

				  mean_vec(i) = mean_vec(i) + meanvector(i);

				  variance_vec(i) = variance_vec(i)*stdvector(i)*stdvector(i);

			  }

			  mean_vec1 = mean_vec1 + basis.col(i)*(mean_vec(i));

			  variance_vec1 = variance_vec1 + (basis.col(i) % basis.col(i))*(variance_vec(i));

		  }


		    double constraint_value = value;

			if (inequalityType == "<") {

				mean_vec1 =  - mean_vec1 + constraint_value;

			}

			if (inequalityType == ">"){

				mean_vec1 =   mean_vec1 - constraint_value;
			}

		    vec probability(constraint_length);   probability.zeros();

		    for (long i = 0; i< constraint_length; i++){

		    	if (variance_vec1(i)!= 0) {

		    		  probability(i) = 1 -cdf(- mean_vec1(i)/sqrt(variance_vec1(i)),0.0,1.0);        // Compute the probability that the constraint function is satisfied.

		    	} else if(mean_vec1(i)> 0)  {

		    		  probability(i) = 1;

		    	}else{
		    		  probability(i) = 0;
		    	}

		     }

		    uword ind = index_min(probability);

		    *mean = mean_vec1(ind);  *variance = variance_vec1(ind);

		    // cout << " predict value is " <<  *mean << endl;
		    // cout << " predict variance is " <<  *variance << endl;


	  }else {

		  surrogate->interpolateWithVariance(x, mean, variance);

		  double meanvalue = surrogate->readOutputMean();
		  double stdvalue  = surrogate->readOutputStd();

		  *mean = (*mean)*stdvalue+meanvalue;  *variance = (*variance)*stdvalue*stdvalue;

	  }

}

double ObjectiveFunction::interpolate(rowvec x) const{

	return surrogate->interpolate(x);

}


void ObjectiveFunction::print(void) const{

	std::cout << "\n#####################################################\n";
	std::cout<<"Objective Function"<<endl;
	std::cout<<"Name: "<<name<<endl;
	std::cout<<"Dimension: "<<dim<<endl;
	std::cout<<"ExecutableName: "<<executableName<<"\n";
	std::cout<<"ExecutablePath: "<<executablePath<<"\n";
	std::cout<<"Output value filename: "<<fileNameOutputValueRead<<"\n";
	std::cout<<"Output gradient filename: "<<fileNameOutputGradRead<<"\n";
	std::cout<<"Design vector filename: "<<fileNameDesignVector<<"\n";

	if(this->ifMarkerIsSet){

		std::cout<<"Read marker: "<<readMarker<<"\n";

	}

	if(ifGradientAvailable){
		std::cout<<"Uses gradient vector: Yes\n";

		if(this->ifAdjointMarkerIsSet){

			std::cout<<"Read marker for gradient: "<<readMarkerAdjoint<<"\n";

		}

	}
	else{
		std::cout<<"Uses gradient vector: No\n";

	}

	std::cout << "#####################################################\n\n";

}


void ObjectiveFunction::printSurrogate(void) const{

	surrogate->printSurrogateModel();

}

bool ObjectiveFunction::checkIfMarkersAreNotSet(void) const{

	if(ifMarkerIsSet == false && ifAdjointMarkerIsSet == false ){

		return true;
	}

	else return false;

}


size_t ObjectiveFunction::isMarkerFound(const std::string &marker, const std::string &inputStr) const{

	assert(isNotEmpty(marker));

	std::string bufferStr(inputStr);
	bufferStr = removeSpacesFromString(bufferStr);
	size_t foundMarkerWithEqualitySign = bufferStr.find(marker+"=");
	size_t foundMarkerWithColonSign    = bufferStr.find(marker+":");



	if(foundMarkerWithEqualitySign != std::string::npos){

		if(foundMarkerWithEqualitySign == 0) return foundMarkerWithEqualitySign;

		else return std::string::npos;
	}

	if(foundMarkerWithColonSign != std::string::npos){

		if(foundMarkerWithColonSign == 0) return foundMarkerWithColonSign;

		else return std::string::npos;
	}


	return std::string::npos;


}



double ObjectiveFunction::getMarkerValue(const std::string &inputStr, size_t foundMarkerPosition) const{


	double result = 0.0;
	std::string bufferStr(inputStr);

	bufferStr = removeSpacesFromString(bufferStr);
	bufferStr.erase(0,foundMarkerPosition+1+readMarker.size());

	return stod(bufferStr);

}

rowvec ObjectiveFunction::getMarkerAdjointValues(const std::string &inputStr, size_t foundMarkerPosition) const{

	rowvec result(dim);
	std::string bufferStr(inputStr);
	bufferStr = removeSpacesFromString(bufferStr);
	bufferStr.erase(0,foundMarkerPosition+1+readMarkerAdjoint.size());
	vec values = getDoubleValuesFromString(bufferStr,',');

	if(values.size() != dim){

		std::cout<<"ERROR: Array size while reading the gradient does not match with the dimension!\n";
		abort();

	}

	for(unsigned int i=0; i<dim;i++){

		result(i) = values(i);

	}

	return result;


}



