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
#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include <cassert>

#include "surrogate_model.hpp"
#include "auxiliary_functions.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "surrogate_model_data.hpp"


#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;



SurrogateModel::SurrogateModel(){

}

SurrogateModel::SurrogateModel(std::string nameInput){


	setName(nameInput);

	filenameDataInput = name +".csv";
	filenameTestResults = name + "_TestResults.csv";

}

void SurrogateModel::setName(std::string nameInput){

	assert(isNotEmpty(nameInput));

	name = nameInput;

}

void SurrogateModel::setGradientsOn(void){
	data.setGradientsOn();
	ifHasGradientData = true;
}

void SurrogateModel::setGradientsOff(void){
	data.setGradientsOff();
	ifHasGradientData = false;
}

bool SurrogateModel::areGradientsOn(void) const{
	return ifHasGradientData;
}


void SurrogateModel::setVectorConstraintOn(void){
	data.setVectorOutputOn();
	ifVectorOutput = true;
}

void SurrogateModel::setConstraintLength(int length){
	data.setConstraintLength(length);
}

void SurrogateModel::setVectorConstraintOff(void){
	data.setVectorOutputOff();
	ifVectorOutput = false;
}

bool SurrogateModel::areVectorConstraint(void) const{
	return ifVectorOutput;
}


void SurrogateModel::setDisplayOn(void){

	data.setDisplayOn();
	output.ifScreenDisplay = true;

}

void SurrogateModel::setDisplayOff(void){

	data.setDisplayOff();
	output.ifScreenDisplay = false;

}


void SurrogateModel::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	data.setBoxConstraints(boxConstraintsInput);

}



void SurrogateModel::setBoxConstraints(vec xmin, vec xmax){

	Bounds boxConstraints(xmin,xmax);
	setBoxConstraints(boxConstraints);

}

void SurrogateModel::setBoxConstraints(double xmin, double xmax){

	Bounds boxConstraints(data.getDimension());
	boxConstraints.setBounds(xmin,xmax);

	setBoxConstraints(boxConstraints);
}


void SurrogateModel::setBoxConstraintsFromData(void){


	data.setBoxConstraintsFromData();


}

std::string SurrogateModel::getNameOfHyperParametersFile(void) const{

	return hyperparameters_filename;

}




std::string SurrogateModel::getNameOfInputFile(void) const{

	return this->filenameDataInput;

}


unsigned int SurrogateModel::getDimension(void) const{

	return data.getDimension();


}

unsigned int SurrogateModel::getNumberOfSamples(void) const{

	return data.getNumberOfSamples();


}

mat SurrogateModel::getRawData(void) const{

	return data.getRawData();


}



void SurrogateModel::readData(void){

	data.readData(filenameDataInput);

	ifDataIsRead = true;

}

void SurrogateModel::assignOutput(int ID){

	data.assignOutput(ID);

	ifDataIsAssigned = true;

}

int SurrogateModel::readRank(void) const{

	return data.getRank();

}

double SurrogateModel::readOutputMean(void) const{

	return data.getOutputMean();

}

double SurrogateModel::readOutputStd(void) const{

	return data.getOutputStd();

}

rowvec SurrogateModel::readOutputMeanVector(void) const{

	return data.getOutputMeanVector();

}

rowvec SurrogateModel::readOutputStdVector(void) const{

	return data.getOutputStdVector();

}


mat SurrogateModel::getPodBasis(void) const{

	return data.getPodBasis();

}

void SurrogateModel::readDataTest(void){

	data.readDataTest(filenameDataInputTest);

}


void SurrogateModel::normalizeData(void){

	assert(ifDataIsRead);

	data.normalizeSampleInputMatrix();
	data.normalizeSampleOutput();

	ifNormalized = true;

}

void SurrogateModel::normalizeDataTest(void){


	data.normalizeSampleInputMatrixTest();
	//data.normalizeSampleOutputMatrixTest();

}


void SurrogateModel::updateAuxilliaryFields(void){


}



double SurrogateModel::calculateInSampleError(void) const{

	assert(ifInitialized);

	double meanSquaredError = 0.0;
	vec y = data.getOutputVector();

	for(unsigned int i=0;i<data.getNumberOfSamples(); i++){

		rowvec xp = data.getRowX(i);

		rowvec x  = data.getRowXRaw(i);

#if 0
		printf("\nData point = %d\n", i+1);
		printf("Interpolation at x:\n");
		x.print();
		printf("xnorm:\n");
		xp.print();
#endif
		double functionValueSurrogate = interpolate(xp);

		double functionValueExact = y(i);

		double squaredError = (functionValueExact-functionValueSurrogate)*(functionValueExact-functionValueSurrogate);

		meanSquaredError+= squaredError;
#if 0
		printf("func_val (exact) = %15.10f, func_val (approx) = %15.10f, squared error = %15.10f\n", functionValueExact,functionValueSurrogate,squaredError);
#endif


	}

	meanSquaredError = meanSquaredError/data.getNumberOfSamples();


	return meanSquaredError;


}

void SurrogateModel::calculateOutSampleError(void){

	assert(ifInitialized);
	assert(ifHasTestData);
	unsigned int numberOfEntries = data.getDimension() + 3;

	testResults = zeros<mat>(NTest, numberOfEntries);

	for(unsigned int i=0;i<NTest;i++){

		rowvec xp = XTest.row(i);

		rowvec x  = XTestraw.row(i);

		double functionValueSurrogate = interpolate(xp);

		double functionValueExact = yTest(i);

		double squaredError = (functionValueExact-functionValueSurrogate)*(functionValueExact-functionValueSurrogate);

		rowvec sample(numberOfEntries);
		copyRowVector(sample,x);
		sample(data.getDimension()) =   functionValueExact;
		sample(data.getDimension()+1) = functionValueSurrogate;
		sample(data.getDimension()+2) = squaredError;

		testResults.row(i) = sample;
	}

}


double SurrogateModel::getOutSampleErrorMSE(void) const{

	assert(testResults.n_rows > 0);

	vec squaredError = testResults.col(data.getDimension()+2);


	return mean(squaredError);



}



void SurrogateModel::saveTestResults(void) const{

	field<std::string> header(testResults.n_cols);

	for(unsigned int i=0; i<data.getDimension(); i++){

		header(i) ="x"+std::to_string(i+1);

	}


	header(data.getDimension())   = "True value";
	header(data.getDimension()+1) = "Estimated value";
	header(data.getDimension()+2) = "Squared Error";

	testResults.save( csv_name(filenameTestResults, header) );


}

void SurrogateModel::visualizeTestResults(void) const{

	std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_Test_Results.py "+ name;

	executePythonScript(python_command);

}





void SurrogateModel::printSurrogateModel(void) const{

	data.print();

}



rowvec SurrogateModel::getRowX(unsigned int index) const{

	return data.getRowX(index);

}


rowvec SurrogateModel::getRowXRaw(unsigned int index) const{

	return data.getRowXRaw(index);
}




void SurrogateModel::setNameOfInputFileTest(string filename){

	assert(isNotEmpty(filename));
	filenameDataInputTest = filename;
}


void SurrogateModel::tryOnTestData(void) const{

	output.printMessage("Trying surrogate model on test data...");

	unsigned int dim = data.getDimension();
	unsigned int numberOfTestSamples = data.getNumberOfSamplesTest();


	if (!ifVectorOutput){

	   unsigned int numberOfEntries = dim + 1;

	   mat results(numberOfTestSamples,numberOfEntries);

	   double std_y =  data.getOutputStd(); double mean_y =  data.getOutputMean();

     //  cout << "The output std " << std_y  <<  endl;
	 //  cout << "The output mean " << mean_y <<  endl;

	   for(unsigned int i=0; i<numberOfTestSamples; i++){

		  rowvec xp = data.getRowXTest(i);
		  rowvec x  = data.getRowXRawTest(i);

		  double fTilde = interpolate(xp);

		  rowvec sample(numberOfEntries);
		  copyRowVector(sample,x);
		  sample(dim)    =  fTilde*std_y + mean_y ;
		  results.row(i) = sample;

	    }

	     output.printMessage("Saving surrogate test results in the file: surrogateTest.csv");

	     saveMatToCVSFile(results,"surrogateTest.csv");

	     output.printMessage("Surrogate test results", results);

	     vec testData = data.getYTest();

	     double mse = mean((testData-results.col(dim))%(testData-results.col(dim)))/var(testData);

         cout << "The relative mean squared error is " << mse <<  endl;

	} else {

		  int rank = readRank();

		  int length = data.getConstraintLength();

		  unsigned int numberOfEntries = dim + length;

		  mat results(numberOfTestSamples,numberOfEntries);

		  mat basis = getPodBasis();
		  vec mean_vec1(length);
		  vec variance_vec1(length);

	 	  rowvec meanvector = readOutputMeanVector();
	 	  rowvec stdvector  = readOutputStdVector();

		  for(unsigned int i=0; i<numberOfTestSamples; i++){

		  	  rowvec xp = data.getRowXTest(i);
		  	  rowvec x  = data.getRowXRawTest(i);

		  	  vec fTilde = interpolate_vec(xp);

			  mean_vec1.zeros();   variance_vec1.zeros();

			  // cout << " mean vector " << meanvector <<  endl;
			  // cout << " std vector "  << stdvector <<  endl;
			  //cout << " rank "  << rank <<  endl;

		  	  for (unsigned int j = 0; j< rank; j++ ){                     // recover the full state solution

		  		 if (stdvector(j) != 0){

		  			 fTilde(j) = fTilde(j)*stdvector(j) + meanvector(j);   // map back to original space

		  		 }else {

		  			 fTilde(j) = fTilde(j) + meanvector(j);                  // map back to original space

		  		 }

		  		 mean_vec1 = mean_vec1 + basis.col(j)*(fTilde(j));

		  	  }

		  	 // mean_vec1 = mean_vec1 % stdvector + meanvector;             // original prediction mean

		  	  rowvec sample(numberOfEntries);
		  	  copyRowVector(sample,x);
		  	  sample.subvec(dim,numberOfEntries-1) = mean_vec1.t() ;
		  	  results.row(i) = sample;

		   }

		         output.printMessage("Saving surrogate test results in the file: surrogateTest.csv");

		 	     saveMatToCVSFile(results,"surrogateTest.csv");

		 	   // output.printMessage("Surrogate test results", results);

		 	     mat testData = data.getY_VectorTest();

		 	     vec mse; mse.zeros(length);

		 	     for (unsigned int i=0; i< length;i++) {

                      if (var(testData.col(i))!= 0){

                    	  mse(i) = mean((testData.col(i)-results.col(dim+i))%((testData.col(i)-results.col(dim+i))))/var(testData.col(i));

                      }
                      else{

                    	  mse(i) = mean((testData.col(i)-results.col(dim+i))%((testData.col(i)-results.col(dim+i))));

                      }

		  }

		         cout << "The total relative mean squared error is " << mean(mse) <<  endl;
	}

}

/* void SurrogateModel::tryOnTestData(void) const{    // Modified by Kai

	output.printMessage("Trying surrogate model on test data...");

	unsigned int dim = data.getDimension();
	unsigned int numberOfEntries = dim + 1;
	unsigned int numberOfTestSamples = data.getNumberOfSamplesTest();


	mat testdata = XTest;

	mat results = interpolate(testdata);

	output.printMessage("Saving surrogate test results in the file: surrogateTest.csv");

	saveMatToCVSFile(results,"surrogateTest.csv");

	output.printMessage("Surrogate test results", results);


	//output.printMessage("Mean squared error", results);


}*/
