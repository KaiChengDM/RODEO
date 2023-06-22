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



#include "surrogate_model_data.hpp"
#include "auxiliary_functions.hpp"
#include "surrogate_model.hpp"
#include <armadillo>
using namespace arma;

SurrogateModelData::SurrogateModelData(){


}

unsigned int SurrogateModelData::getNumberOfSamples(void) const{


	return numberOfSamples;
}

unsigned int SurrogateModelData::getNumberOfSamplesTest(void) const{


	return numberOfTestSamples;
}


unsigned int SurrogateModelData::getDimension(void) const{


	return dimension;
}


void SurrogateModelData::setDimension(unsigned int value){

	dimension = value;

}


mat SurrogateModelData::getRawData(void) const{

	return rawData;


}


void SurrogateModelData::setDisplayOn(void){

	outputToScreen.ifScreenDisplay = true;
	outputToScreen.printMessage("Setting display on for the SurrogateModelData...");

}

void SurrogateModelData::setDisplayOff(void){

	outputToScreen.ifScreenDisplay = false;


}

void SurrogateModelData::setGradientsOn(void){
	ifDataHasGradients = true;
}


void SurrogateModelData::setGradientsOff(void){
	ifDataHasGradients = false;
}

void SurrogateModelData::setVectorOutputOn(void){

	ifVectorOutput = true;

}

void SurrogateModelData::setVectorOutputOff(void){

	ifVectorOutput = false;

}

void SurrogateModelData::setConstraintLength(int length){

	constraintLength = length;
}


int SurrogateModelData::getConstraintLength(void) const {

	return constraintLength;

}

void SurrogateModelData::readData(string inputFilename){

	assert(isNotEmpty(inputFilename));

	outputToScreen.printMessage("Loading data from the file: " + inputFilename);

	cout << " Loading data from the file: " << inputFilename << endl;

	bool status = rawData.load(inputFilename.c_str(), csv_ascii);

	if(status == true)
	{
		outputToScreen.printMessage("Data input is done...");

	}
	else
	{
		outputToScreen.printErrorMessageAndAbort("Problem with data the input (cvs ascii format), cannot read: " + inputFilename);

	}


	numberOfSamples = rawData.n_rows;
	outputToScreen.printMessage("Number of samples = ", numberOfSamples);
	//outputToScreen.printMessage("Raw data = ", rawData);

	assignDimensionFromData();

	assignSampleInputMatrix();

	if (ifVectorOutput){

		assignSampleOutputMatrix();

	}
	else{
		assignSampleOutputVector();
	}

	assignGradientMatrix();

}

void SurrogateModelData::readDataTest(string inputFilename){

	assert(isNotEmpty(inputFilename));

	outputToScreen.printMessage("Loading test data from the file: " + inputFilename);

	bool status = XrawTest.load(inputFilename.c_str(), csv_ascii);

	if(status == true)
	{
		outputToScreen.printMessage("Data input is done...");

	}
	else
	{
		outputToScreen.printErrorMessageAndAbort("Problem with data the input (cvs ascii format), cannot read: " + inputFilename);

	}

	numberOfTestSamples = XrawTest.n_rows;

	outputToScreen.printMessage("Number of test samples = ", numberOfTestSamples);

	XTest = XrawTest.cols(0,dimension-1);

	if (ifVectorOutput){
		Y_vectTest.zeros(numberOfTestSamples,dimension+constraintLength-1);
		Y_vectTest =  XrawTest.cols(dimension,dimension+constraintLength-1);

	}else {

		YTest = XrawTest.col(dimension);

	}
}



void SurrogateModelData::assignDimensionFromData(void){


	unsigned int dimensionOfTrainingData;

    // if()
	// cout << "if gradient " << ifDataHasGradients << endl;
	// cout << "if vector constraint " << ifVectorOutput << endl;

	if(ifDataHasGradients){

		dimensionOfTrainingData = (rawData.n_cols-1)/2;
	}

	else{

		if (ifVectorOutput){

			dimensionOfTrainingData =  rawData.n_cols-constraintLength;

        }else{

        	dimensionOfTrainingData =  rawData.n_cols-1;

        }
	}


	if(dimension > 0 && dimensionOfTrainingData!= dimension){

		outputToScreen.printErrorMessageAndAbort("Dimension of the training data does not match with the specified dimension!");
	}

	dimension = dimensionOfTrainingData;

	outputToScreen.printMessage("Dimension of the problem is identified as ", dimension);

}

void SurrogateModelData::assignSampleInputMatrix(void){

	assert(dimension>0);
	assert(numberOfSamples>0);

	X = rawData.submat(0,0,numberOfSamples-1, dimension-1);

	Xraw = X;

}

void SurrogateModelData::assignSampleOutputVector(void){

	assert(dimension>0);
	assert(numberOfSamples>0);

	y = rawData.col(dimension);

}

void SurrogateModelData::assignSampleOutputMatrix(void){

	assert(dimension>0);
	assert(numberOfSamples>0);

	mat Y = rawData.submat(0,dimension,numberOfSamples-1, dimension+constraintLength-1);

	y_vec = Y.t();
}

void SurrogateModelData::assignOutput(int ID){

	y = pod_basiscoefficient.col(ID);

}

int SurrogateModelData::getRank(void) const{

	return rank;
}

void SurrogateModelData::pod_ROM(void){

	mat U;  vec s;  mat V;

	svd_econ(U, s, V, y_vec);

	eigenvalue = s%s;

	for (unsigned int i=0; i<numberOfSamples; i++){

	   double ratio = sum(eigenvalue.subvec(0,i))/sum(eigenvalue);


        if (ratio > threshold){
        	rank = i+1;
        	break;
        }

	 }

	pod_basis = U.cols(0,rank-1);

    pod_basiscoefficient = y_vec.t()*pod_basis;

    mean_basiscoefficient = mean(pod_basiscoefficient,0);

    std_basiscoefficient  = stddev(pod_basiscoefficient,0,0);


    for (unsigned int i=0; i < rank; i++){

    	  if (std_basiscoefficient(i) != 0) {

    	    pod_basiscoefficient.col(i) = (pod_basiscoefficient.col(i) - mean_basiscoefficient(i))/std_basiscoefficient(i);  // normalize the POD coefficient

    	  }
    	  else{

    	    pod_basiscoefficient.col(i) = pod_basiscoefficient.col(i) - mean_basiscoefficient(i);  // normalize the POD coefficient

    	   }
    }

  //   cout << "mean_basiscoefficient " << mean_basiscoefficient << endl;
  //   cout << " std_basiscoefficient " << std_basiscoefficient << endl;

	cout << "The proper orthogonal decomposition mode number is " << rank << endl;

}


void SurrogateModelData::assignGradientMatrix(void){

	assert(dimension>0);
	assert(numberOfSamples>0);


	if(ifDataHasGradients){

		assert(rawData.n_cols > dimension+1);

		gradient = rawData.submat(0, dimension+1, numberOfSamples - 1, 2*dimension);

	}


}


rowvec SurrogateModelData::getRowGradient(unsigned int index) const{

	return gradient.row(index);

}


rowvec SurrogateModelData::getRowRawData(unsigned int index) const{

	return rawData.row(index);

}



rowvec SurrogateModelData::getRowX(unsigned int index) const{

	assert(index < X.n_rows);

	return X.row(index);

}

rowvec SurrogateModelData::getRowXTest(unsigned int index) const{

	assert(index < XTest.n_rows);

	return XTest.row(index);

}

vec SurrogateModelData::getYTest(void) const{   // Modified by Kai

	return YTest;

}

mat SurrogateModelData::getY_VectorTest(void) const{   // Modified by Kai

	return Y_vectTest;

}


rowvec SurrogateModelData::getRowXRaw(unsigned int index) const{

	assert(index < Xraw.n_rows);

	return Xraw.row(index);
}

rowvec SurrogateModelData::getRowXRawTest(unsigned int index) const{

	assert(index < XrawTest.n_rows);

	return XrawTest.row(index);
}


vec SurrogateModelData::getOutputVector(void) const{

	return y;

}

double SurrogateModelData::getOutputMean(void) const{

	return mean_y;

}

double SurrogateModelData::getOutputStd(void) const{

	return std_y;

}

rowvec SurrogateModelData::getOutputMeanVector(void) const{

	// return mean_y_vec;
	return mean_basiscoefficient;

}

rowvec SurrogateModelData::getOutputStdVector(void) const{

	//return std_y_vec;
	return std_basiscoefficient;
}


mat SurrogateModelData::getPodBasis(void) const{

	return  pod_basis;

}

mat SurrogateModelData::getPodBasisCoefficients(void) const{

	return pod_basiscoefficient;

}

mat SurrogateModelData::getInputMatrix(void) const{

	return X;

}


double SurrogateModelData::getMinimumOutputVector(void) const{

	return min(y);
}



mat SurrogateModelData::getGradientMatrix(void) const{

	return gradient;

}




void SurrogateModelData::normalizeSampleInputMatrix(void){

	assert(X.n_rows ==  numberOfSamples);
	assert(X.n_cols ==  dimension);

	assert(boxConstraints.getDimension() == dimension);
	assert(boxConstraints.areBoundsSet());

	outputToScreen.printMessage("Normalizing and scaling the sample input matrix...");

	mat XNormalized = X;
	vec xmin = boxConstraints.getLowerBounds();
	vec xmax = boxConstraints.getUpperBounds();
	vec deltax = xmax - xmin;


	for(unsigned int i=0; i<numberOfSamples;i++){

		for(unsigned int j=0; j<dimension;j++){

			XNormalized(i,j) = (X(i,j) - xmin(j))/deltax(j);

		}

	}

	X = (1.0/dimension)*XNormalized;
	ifDataIsNormalized = true;
}

void SurrogateModelData::normalizeSampleOutput(void){

	assert(X.n_rows ==  numberOfSamples);
	assert(X.n_cols ==  dimension);

	assert(boxConstraints.getDimension() == dimension);

	assert(boxConstraints.areBoundsSet());

	// mean_y_vec.zeros(constraintLength); std_y_vec.ones(constraintLength);

	if (ifVectorOutput){

		/*mean_y_vec = mean(y_vec,1); std_y_vec = stddev(y_vec,0,1);

		for (unsigned int i=0; i<  mean_y_vec.size(); i++){

			if (std_y_vec(i)!=0) {

				y_vec.row(i) = (y_vec.row(i) -  mean_y_vec (i))/std_y_vec(i);

			} else{

			    y_vec.row(i) = y_vec.row(i) -  mean_y_vec (i);

			}

		}*/

		pod_ROM();  // proper orthogonal decomposition based reduced order model

	}else {

		mean_y = mean(y);  std_y = stddev(y);

	    if (std_y == 0){

	    	printf("ERROR: The output samples are the same!\n");
	    	abort();

	    }else {

	    	y =  (y -mean_y) / std_y;

	    }

	}

	ifOutputIsNormalized = true;

}


void SurrogateModelData::normalizeSampleInputMatrixTest(void){

	assert(boxConstraints.areBoundsSet());

	outputToScreen.printMessage("Normalizing and scaling the sample input matrix for test...");

	mat XNormalized = XTest;
	vec xmin = boxConstraints.getLowerBounds();
	vec xmax = boxConstraints.getUpperBounds();
	vec deltax = xmax - xmin;


	for(unsigned int i=0; i<numberOfTestSamples;i++){

		for(unsigned int j=0; j<dimension;j++){

			XNormalized(i,j) = (XTest(i,j) - xmin(j))/deltax(j);


		}

	}

	XTest = (1.0/dimension)*XNormalized;
	//XTest = XNormalized;

}

/*void SurrogateModelData::normalizeSampleOutputTest(void){

	assert(X.n_rows ==  numberOfSamples);
	assert(X.n_cols ==  dimension);

	assert(boxConstraints.getDimension() == dimension);

	assert(boxConstraints.areBoundsSet());

	// mean_y_vec.zeros(constraintLength); std_y_vec.ones(constraintLength);

	if (ifVectorOutput){

		pod_ROM();  // proper orthogonal decomposition based reduced order model

	}else {

		mean_y = mean(y);  std_y = stddev(y);
        y =  (y -mean_y) / std_y;
	}

	ifOutputIsNormalized = true;

}*/


void SurrogateModelData::setBoxConstraints(Bounds boxConstraintsInput){

	assert(boxConstraintsInput.areBoundsSet());

	boxConstraints = boxConstraintsInput;

}

void SurrogateModelData::setBoxConstraintsFromData(void){


	outputToScreen.printMessage("setting box constraints from the training data...");

	vec maxofX(dimension);
	vec minofX(dimension);



	for(unsigned int i=0; i<dimension; i++){

		minofX(i) = min(Xraw.col(i));
		maxofX(i) = max(Xraw.col(i));


	}



	boxConstraints.setBounds(minofX,maxofX);

	outputToScreen.printMessage("lower bounds", minofX);
	outputToScreen.printMessage("upper bounds", maxofX);



}


Bounds SurrogateModelData::getBoxConstraints(void) const{

	return boxConstraints;

}



bool SurrogateModelData::isDataNormalized(void) const{

	return ifDataIsNormalized;

}

bool SurrogateModelData::isVectorOutput(void) const{

	return ifVectorOutput;

}


void SurrogateModelData::print(void) const{

	printMatrix(rawData,"raw data");
	printMatrix(X,"sample input matrix");
	printVector(y,"sample output vector"); // Modified by Kai

	if(ifDataHasGradients){

		printMatrix(gradient,"sample gradient matrix");

	}

}


//void SurrogateModel::checkRawData(void) const{
//
//	for(unsigned int i=0; i<numberOfSamples; i++){
//
//		rowvec sample1 = rawData.row(i);
//
//		for(unsigned int j=i+1; j<numberOfSamples; j++){
//
//			rowvec sample2 = rawData.row(j);
//
//			if(checkifTooCLose(sample1, sample2)) {
//
//				printf("ERROR: Two samples in the training data are too close to each other!\n");
//
//				abort();
//			}
//		}
//	}
//
//
//
//}
