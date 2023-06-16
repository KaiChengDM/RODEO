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

#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include <cassert>

#include "linear_regression.hpp"
#include "auxiliary_functions.hpp"
#include "random_functions.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "gek.hpp"
#include "correlation_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;

/* global variables */

int total_number_of_function_evals_GEK;
double population_overall_max_GEK = -10E14;
int population_overall_max_tread_id_GEK = -1;

GEKModel::GEKModel():SurrogateModel(){}


GEKModel::GEKModel(std::string nameInput):SurrogateModel(nameInput){

	modelID = GRADIENT_ENHANCED_KRIGING;
	setName(nameInput);
	setNameOfHyperParametersFile(nameInput);
	maxNumberOfTrainingIterations = 10000;

}


void GEKModel::setNameOfInputFile(std::string filename){

	assert(!filename.empty());
	filenameDataInput = filename;

}

void GEKModel::setNumberOfTrainingIterations(unsigned int nIters){

	numberOfTrainingIterations = nIters;

}

void GEKModel::setNameOfHyperParametersFile(std::string filename){

	assert(!filename.empty());
	hyperparameters_filename = filename;

}



void GEKModel::initializeSurrogateModel(void){

	unsigned int dim = data.getDimension();
	unsigned int numberOfSamples = data.getNumberOfSamples();

	printf("Initializing settings for the GEK model...\n");

	modelID = GRADIENT_ENHANCED_KRIGING;
	ifHasGradientData = true;

	readData();
	normalizeData();

	numberOfHyperParameters = dim;

	GEK_weights = zeros<vec>(numberOfHyperParameters);

	/* regularization term */

	epsilonGEK = 0.0;

	/* check if two sample are too close to each other */

	for(unsigned int i=0; i<numberOfSamples; i++){

		rowvec sample1 = data.getRowX(i);


		for(unsigned int j=i+1; j<numberOfSamples; j++){

			rowvec sample2 = data.getRowX(j);

			if(checkifTooCLose(sample1, sample2)) {

				printf("ERROR: Two samples in the training data are too close to each other!\n");
				abort();
			}
		}
	}

	sigmaSquared = 0.0;
	beta0        = 0.0;

	correlationMatrixDot   = zeros(numberOfSamples*(dim+1),numberOfSamples*(dim+1));
	upperDiagonalMatrixDot = zeros<mat>(numberOfSamples*(dim+1),numberOfSamples*(dim+1));
	R_inv_ys_min_beta      = zeros<vec>(numberOfSamples*(dim+1));
	R_inv_F                = zeros<vec>(numberOfSamples*(dim+1));
	vectorOfF              = zeros<vec>(numberOfSamples*(dim+1));

	for(unsigned int i=0; i<numberOfSamples; i++) {

		vectorOfF(i)=1.0;
	}


	yGEK = zeros<vec>(numberOfSamples*(dim+1));

	/* first N entries are the functional values */

	vec y = data.getOutputVector();

	for(unsigned int i=0; i<numberOfSamples; i++){

		yGEK(i) = y(i);

	}

	/* rest entries are the gradients (scaled to the feature space) */

	mat gradientData = data.getGradientMatrix();

	Bounds boxConstraints = data.getBoxConstraints();

	double std_y =  data.getOutputStd();

	for(unsigned int i=0; i<dim; i++){

		vec gradx = gradientData.col(i);

		for(unsigned int j=0; j<numberOfSamples; j++){

			double xmin = boxConstraints.getLowerBound(i);
			double xmax = boxConstraints.getUpperBound(i);

			yGEK(numberOfSamples+i*numberOfSamples+j) = gradx(j)*( xmax - xmin )*dim/std_y;
			//yGEK(numberOfSamples+i*numberOfSamples+j) = gradx(j)*( xmax - xmin );


		}
	}

#if 0
	printVector(y,"y");
	printMatrix(gradientData,"gradientData");
	printVector(yGEK,"yGEK");
#endif

	ifInitialized = true;

	std::cout << "GEK model initialization is done...\n";


}

void GEKModel::printSurrogateModel(void) const{

	data.print();

	printVector(GEK_weights,"GEK_weights");

}

void GEKModel::printHyperParameters(void) const{

	printVector(GEK_weights,"GEK_weights");


}

void GEKModel::saveHyperParameters(void) const{


}

void GEKModel::loadHyperParameters(void){


}

void GEKModel::train(void){

	if(!ifInitialized){
		initializeSurrogateModel();
	}

	unsigned int dim = data.getDimension();

	vec hyper_l = 0.001*dim*ones(dim,1);
	vec hyper_u = 10*dim*ones(dim,1);

	num = 10;                       // Multiple starts

	clock_t start, finish;

	start = clock();

	boxmin(hyper_l,hyper_u,num);    // Hooke Jeeves algorithm for hyper-parameter optimization

	finish = clock();

	cout << "GEK model training time is " << (double)(finish-start)/CLOCKS_PER_SEC  << endl;

	GEK_weights = getOptimalTheta();

	likelihood_optimal  = getOptimalLikelihood();

	// cout << "GEK model hyper-paraemter is " << GEK_weights  << endl;
	// cout << "GEK model likelihood is " << likelihood_optimal << endl;

#if 0
	printVector(GEK_weights,"GEK_weights");
#endif

	updateAuxilliaryFields();

}

double GEKModel::likelihood_function(vec theta){

	unsigned int dim = data.getDimension();
	unsigned int N = data.getNumberOfSamples();
	int mn = N*(dim+1);

	mat X = data.getInputMatrix();

	correlationMatrixDot = correlationfunction.corrbiquadspline_gekriging(X,theta);

	upperDiagonalMatrixDot = chol(correlationMatrixDot);

	long double logdetR = 2*sum(log(diagvec(upperDiagonalMatrixDot)));

	vec R_inv_ys(mn); R_inv_ys.fill(0.0);

	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_ys, yGEK);         /* solve R x = ys */

	R_inv_F = zeros(mn);

	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_F, vectorOfF);      /* solve R x = F */

	beta0 = (1.0/dot(vectorOfF,R_inv_F)) * (dot(vectorOfF,R_inv_ys));

	vec ys_min_betaF = yGEK - beta0*vectorOfF;

	/* solve R x = ys-beta0*I */

	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_ys_min_beta , ys_min_betaF);

	sigmaSquared = (1.0 / (mn)) * dot(ys_min_betaF, R_inv_ys_min_beta);

    likelihood = mn * log(sigmaSquared) + logdetR;

    return likelihood;

}


 double GEKModel::interpolate(rowvec x ) const {


	vec r = computeCorrelationVectorDot(x);
#if 0
	printVector(r,"r");
	cout<<"beta0 ="<<beta0<<"\n";
#endif


	double fGEK = beta0 + dot(r,R_inv_ys_min_beta);

	return fGEK;

}

 vec GEKModel::interpolate_vec(rowvec x ) const {

	 cout << "ERROR: interpolate for vector output has not been utilized in GEK \n";
	 abort();

 }


/* mat GEKModel::interpolate_all(mat xtest) {

	mat X = data.getInputMatrix();
	vec theta = GEK_weights;

	mat r = correlationfunction.corrbiquadspline_gekriging_vec(xtest,X,theta);

   // mat r = computeCorrelationVectorDot(x);
   //vec fGEK = beta0 + dot(r,R_inv_ys_min_beta);

	vec fGEK;

	for (unsigned int k = 0; k < dim; k++) {

		fGEK(k) = beta0 + dot(r.row(k),R_inv_ys_min_beta);

	}

	return fGEK;
}*/

void GEKModel::interpolateWithVariance(rowvec xp,double *ftildeOutput,double *sSqrOutput) const {

	unsigned int dim             = data.getDimension();
	unsigned int numberOfSamples = data.getNumberOfSamples();

	*ftildeOutput =  interpolate(xp);

	vec R_inv_r(numberOfSamples*(dim+1));

	vec r = computeCorrelationVectorDot(xp);

	/* solve the linear system R x = r by Cholesky matrices U and L*/

	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_r, r);

	*sSqrOutput = sigmaSquared*( 1.0 - dot(r,R_inv_r)+ ( pow( (dot(r,R_inv_F) -1.0 ),2.0)) / (dot(vectorOfF,R_inv_F) ) );

}


void GEKModel::interpolateWithVariance_vec(rowvec xp,vec &f_tilde, vec &ssqr) const{

	cout << "ERROR: interpolateWithVariance for vector output has not been utilized in GEK \n";
	abort();

}

/*
 * derivative of R(x^i,x^j) w.r.t. x^i_k (for GEK)
 * /

/* double GEKModel::computedR_dxi(rowvec x_i, rowvec x_j,int k) const{

	vec theta = GEK_weights;
	double result;

	double R = computeCorrelation(x_i, x_j, theta);
	result= -2.0*theta(k)* (x_i(k)-x_j(k))* R;
	return result;
}*/

 double GEKModel::computedR_dxi(rowvec x_i, rowvec x_j,int k) const{

	vec theta = GEK_weights;
	double R  = computeCorrelation(x_i, x_j, theta);

	double result;
	double xi;
	double ui;

	// result= -2.0*theta(k)* (x_i(k)-x_j(k))* R;

	xi = fabs(x_i(k)-x_j(k))*theta(k);
	ui = sign(x_i(k)-x_j(k))*theta(k);

	double xi2 = xi*xi; double xi3 = xi2*xi;  double xi4 = xi3*xi;

	if (xi <= 0.4)
		 { result = -ui*(-30*xi + 105*xi2 - 195.0/2*xi3)/(1 - 15*xi2 + 35*xi3 - 195.0/8*xi4)*R;}
    else if (xi < 1)
		 { result = -ui*(-20.0/3 + 20*xi- 20*xi2 + 20.0/3*xi3)/(5/3 - 20.0/3*xi + 10*xi2 - 20.0/3*xi3 + 5.0/3*xi4)*R;}
	else
		 { result = 0;}

	return result;

}

/* double GEKModel::computedR_dxi(rowvec x_i, rowvec x_j,int k) const{

	vec theta = GEK_weights;
	double result;
    double xi;
    double ui;
	double R = computeCorrelation(x_i, x_j, theta);

	// result= -2.0*theta(k)* (x_i(k)-x_j(k))* R;

	xi = fabs(x_i(k)-x_j(k))*theta(k);
	ui = sign(x_i(k)-x_j(k))*theta(k);

	if (xi <= 0.4)

		 { result = -ui*(-30*xi + 105*pow(xi,2) - 195.0/2*pow(xi,3))/(1 - 15*pow(xi,2) + 35*pow(xi,3) - 195.0/8*pow(xi,4))*R;}

    else if (xi < 1)

		 { result = -ui*(-20.0/3 + 20*xi- 20*pow(xi,2) + 20.0/3*pow(xi,3))/(5/3 - 20.0/3*xi + 10*pow(xi,2) - 20.0/3*pow(xi,3) + 5.0/3*pow(xi,4))*R;}

	else

		 { result = 0;}

	return result;
}*/

/*
 * derivative of R(x^i,x^j) w.r.t. x^j_k (for GEK)
 * /

/* double GEKModel::computedR_dxj(rowvec x_i, rowvec x_j, int k) const {

	vec theta = GEK_weights;
	double result = 0.0;
	double R = computeCorrelation(x_i, x_j, theta);

	result= 2.0*theta(k)* (x_i(k)-x_j(k))* R;

	return result;
}*/

double GEKModel::computedR_dxj(rowvec x_i, rowvec x_j, int k) const {

	vec theta = GEK_weights;
	double R = computeCorrelation(x_i, x_j, theta);

    double xi;
    double ui;
	double result;

    // result= 2.0*theta(k)* (x_i(k)-x_j(k))* R;

	xi = fabs(x_i(k)-x_j(k))*theta(k);
	ui = sign(x_i(k)-x_j(k))*theta(k);

	double xi2 = xi*xi; double xi3 = xi2*xi;  double xi4 = xi3*xi;

	if (xi <= 0.4)
       { result = -ui*(-30*xi + 105*xi2 - 195.0/2*xi3)/(1 - 15*xi2 + 35*xi3 - 195.0/8*xi4)*R;}
    else if (xi < 1)
	   { result = -ui*(-20.0/3 + 20*xi- 20*xi2 + 20.0/3*xi3)/(5.0/3 - 20.0/3*xi + 10*xi2 - 20.0/3*xi3 + 5.0/3*xi4)*R;}
	else
	   { result = 0;}

	return result;

}

/*
double GEKModel::computedR_dxj(rowvec x_i, rowvec x_j, int k) const {

	vec theta     = GEK_weights;
	double result = 0.0;
	double R      = computeCorrelation(x_i, x_j, theta);
    double xi;
    double ui;

    // result= 2.0*theta(k)* (x_i(k)-x_j(k))* R;

	xi  = fabs(x_i(k)-x_j(k))*theta(k);
	ui  = sign(x_i(k)-x_j(k))*theta(k);

	if (xi <= 0.4)

       { result = -ui*(-30*xi + 105*pow(xi,2) - 195.0/2*pow(xi,3))/(1 - 15*pow(xi,2) + 35*pow(xi,3) - 195.0/8*pow(xi,4))*R;}

    else if (xi < 1)

	   { result = -ui*(-20.0/3 + 20*xi- 20*pow(xi,2) + 20.0/3*pow(xi,3))/(5.0/3 - 20.0/3*xi + 10*pow(xi,2) - 20.0/3*pow(xi,3) + 5.0/3*pow(xi,4))*R;}

	else

	   { result = 0;}

	return result;
}*/

/*
 * second derivative of R(x^i,x^j) w.r.t. x^i_l and x^j_k (hand derivation)
 * (for GEK)
 * /

/* double GEKModel::computedR_dxi_dxj(rowvec x_i, rowvec x_j, int l,int k) const{

	double dx;
	vec theta = GEK_weights;
	double R = computeCorrelation(x_i, x_j, theta);

	if (k == l){

		dx = 2.0*theta(k)*(-2.0*theta(k)*pow((x_i(k)-x_j(k)),2.0)+1.0)*R;
	}
	if (k != l) {

		dx = -4.0*theta(k)*theta(l)*(x_i(k)-x_j(k))*(x_i(l)-x_j(l))*R;
	}

	return dx;
}*/

/*
double GEKModel::computedR_dxi_dxj(rowvec x_i, rowvec x_j, int l,int k) const{

	double dx;
	double xi;
	double xi1;
	double ui1;
	double xi2;
	double ui2;

	vec theta = GEK_weights;

	double R = computeCorrelation(x_i, x_j, theta);

	if (k == l){

   // dx = 2.0*theta(k)*(-2.0*theta(k)*pow((x_i(k)-x_j(k)),2.0)+1.0)*R;

	 xi = fabs(x_i(k)-x_j(k))*theta(k);
	 if (xi <= 0.4)
		 { dx = -(-30 + 210*xi - 585.0/2*pow(xi,2))*pow(theta(k),2)/(1-15*pow(xi,2) + 35*pow(xi,3)-195.0/8*pow(xi,4))*R;}
	 else if (xi < 1)
		 { dx= -(20 - 40*xi + 20*pow(xi,2))*pow(theta(k),2)/(5.0/3 - 20.0/3*xi + 10*pow(xi,2) - 20.0/3*pow(xi,3) + 5.0/3*pow(xi,4))*R;}
     else
		 { dx = 0;}

	}
	if (k != l) {

		// dx = -4.0*theta(k)*theta(l)*(x_i(k)-x_j(k))*(x_i(l)-x_j(l))*R;

		xi1  = fabs(x_i(k)-x_j(k))*theta(k);
		ui1  = sign(x_i(k)-x_j(k))*theta(k);
		xi2  = fabs(x_i(l)-x_j(l))*theta(l);
		ui2  = sign(x_i(l)-x_j(l))*theta(l);

		if (xi1 <= 0.4 && xi2 <= 0.4)
			 { dx = -(ui1*(-30*xi1 + 105*pow(xi1,2) - 195.0/2*pow(xi1,3))*ui2*(-30*xi2 + 105*pow(xi2,2) - 195.0/2*pow(xi2,3)))/((1 - 15*pow(xi1,2) + 35*pow(xi1,3) - 195.0/8*pow(xi1,4))*(1 - 15*pow(xi2,2) + 35*pow(xi2,3) - 195.0/8*pow(xi2,4)))*R;}
		else if (xi1 < 1 && xi2 <= 0.4)
		     { dx = -(ui1*(-20.0/3 + 20*xi1- 20*pow(xi1,2) + 20.0/3*pow(xi1,3))*ui2*(-30*xi2 + 105*pow(xi2,2) - 195.0/2*pow(xi2,3)))/((5.0/3 - 20.0/3*xi1 + 10*pow(xi1,2) - 20.0/3*pow(xi1,3) + 5.0/3*pow(xi1,4))*(1 - 15*pow(xi2,2) + 35*pow(xi2,3) - 195.0/8*pow(xi2,4)))*R;}
		else if(xi1 < 1 && xi2 < 1 )
		     { dx = -(ui1*(-20.0/3 + 20*xi1- 20*pow(xi1,2) + 20.0/3*pow(xi1,3))*ui2*(-20.0/3 + 20*xi2- 20*pow(xi2,2) + 20.0/3*pow(xi2,3)))/((5/3 - 20.0/3*xi1 + 10*pow(xi1,2) - 20.0/3*pow(xi1,3) + 5.0/3*pow(xi1,4))*(5.0/3 - 20.0/3*xi2 + 10*pow(xi2,2) - 20.0/3*pow(xi2,3) + 5.0/3*pow(xi2,4)))*R;}
		else if (xi1 < 0.4 && xi2 < 1 )
		     { dx = -(ui1*(-30*xi1 + 105*pow(xi1,2) - 195.0/2*pow(xi1,3))*ui2*(-20.0/3 + 20*xi2- 20*pow(xi2,2) + 20.0/3*pow(xi2,3)))/((1 - 15*pow(xi1,2) + 35*pow(xi1,3) - 195.0/8*pow(xi1,4))*(5.0/3 - 20.0/3*xi2 + 10*pow(xi2,2) - 20.0/3*pow(xi2,3) + 5.0/3*pow(xi2,4)))*R;}
		else
			 { dx = 0;}
	}

	return dx;
}*/


double GEKModel::computedR_dxi_dxj(rowvec x_i, rowvec x_j, int l,int k) const{

	double dx;
	double xi;
	double xi1;
	double ui1;
	double xi2;
	double ui2;

	vec theta = GEK_weights;
	double R  = computeCorrelation(x_i, x_j, theta);

	if (k == l){

    // dx = 2.0*theta(k)*(-2.0*theta(k)*pow((x_i(k)-x_j(k)),2.0)+1.0)*R;

	 xi = fabs(x_i(k)-x_j(k))*theta(k);

	 double xi_2 = xi*xi; double xi_3 = xi_2*xi;  double xi_4 = xi_3*xi;

	 if (xi <= 0.4)
		 { dx = -(-30 + 210*xi - 585.0/2*xi_2)*pow(theta(k),2)/(1-15*xi_2 + 35*xi_3-195.0/8*xi_4)*R;}
	 else if (xi < 1)
		 { dx = -(20 - 40*xi + 20*xi_2)*pow(theta(k),2)/(5.0/3 - 20.0/3*xi + 10*xi_2 - 20.0/3*xi_3 + 5.0/3*xi_4)*R;}
     else
		 { dx = 0;}

	}
	if (k != l) {

		// dx = -4.0*theta(k)*theta(l)*(x_i(k)-x_j(k))*(x_i(l)-x_j(l))*R;

		xi1  = fabs(x_i(k)-x_j(k))*theta(k);
		ui1  = sign(x_i(k)-x_j(k))*theta(k);
		xi2  = fabs(x_i(l)-x_j(l))*theta(l);
		ui2  = sign(x_i(l)-x_j(l))*theta(l);

	    double xi1_2 = xi1*xi1; double xi1_3 = xi1_2*xi1;  double xi1_4 = xi1_3*xi1;
	    double xi2_2 = xi2*xi2; double xi2_3 = xi2_2*xi2;  double xi2_4 = xi2_3*xi2;

		if (xi1 <= 0.4 && xi2 <= 0.4)
			 { dx = -(ui1*(-30*xi1 + 105*xi1_2 - 195.0/2*xi1_3)*ui2*(-30*xi2 + 105*xi2_2 - 195.0/2*xi2_3))/((1 - 15*xi1_2 + 35*xi1_3 - 195.0/8*xi1_4)*(1 - 15*xi2_2 + 35*xi2_3 - 195.0/8*xi2_4))*R;}
		else if (xi1 < 1 && xi2 <= 0.4)
		     { dx = -(ui1*(-20.0/3 + 20*xi1- 20*xi1_2 + 20.0/3*xi1_3)*ui2*(-30*xi2 + 105*xi2_2 - 195.0/2*xi2_3))/((5.0/3 - 20.0/3*xi1 + 10*xi1_2 - 20.0/3*xi1_3 + 5.0/3*xi1_4)*(1 - 15*xi2_2 + 35*xi2_3 - 195.0/8*xi2_4))*R;}
		else if(xi1 < 1 && xi2 < 1 )
		     { dx = -(ui1*(-20.0/3 + 20*xi1- 20*xi1_2 + 20.0/3*xi1_3)*ui2*(-20.0/3 + 20*xi2- 20*xi2_2 + 20.0/3*xi2_3))/((5/3 - 20.0/3*xi1 + 10*xi1_2 - 20.0/3*xi1_3 + 5.0/3*xi1_4)*(5.0/3 - 20.0/3*xi2 + 10*xi2_2 - 20.0/3*xi2_3 + 5.0/3*xi2_4))*R;}
		else if (xi1 < 0.4 && xi2 < 1 )
		     { dx = -(ui1*(-30*xi1 + 105*xi1_2 - 195.0/2*xi1_3)*ui2*(-20.0/3 + 20*xi2- 20*xi2_2 + 20.0/3*xi2_3))/((1 - 15*xi1_2 + 35*xi1_3 - 195.0/8*xi1_4)*(5.0/3 - 20.0/3*xi2 + 10*xi2_2 - 20.0/3*xi2_3 + 5.0/3*xi2_4))*R;}
		else
			 { dx = 0;}
	}

	return dx;
}

/* double GEKModel::computeCorrelation(rowvec x_i, rowvec x_j, vec theta) const {

	unsigned int dim = data.getDimension();
	double sum = 0.0;
	for (unsigned int k = 0; k < dim; k++) {

		sum += theta(k) * pow(fabs(x_i(k) - x_j(k)), 2.0);

	}

	return exp(-sum);
}

/* double GEKModel::computeCorrelation(rowvec x_i, rowvec x_j, vec theta) const {

	unsigned int dim = data.getDimension();
	double sum = 0.0;
	double xi = 0.0;
	vec ss(dim);

	for (unsigned int k = 0; k < dim; k++) {

		  xi = fabs(x_i(k) - x_j(k))*theta(k);

		  if (xi <= 0.4)
			  {ss(k) = 1 - 15*pow(xi,2) + 35*pow(xi,3) - 195.0/8*pow(xi,4);}
		  else if (xi < 1)
			  {ss(k) = 5.0/3 - 20.0/3*xi + 10*pow(xi,2) - 20.0/3*pow(xi,3) + 5.0/3*pow(xi,4);}
		  else
			  {ss(k) = 0;}
        }

	return prod(ss);
}*/


double GEKModel::computeCorrelation(rowvec x_i, rowvec x_j, vec theta) const {

	unsigned int dim = data.getDimension();
	double sum = 0.0;
	double xi  = 0.0;
	vec ss(dim);

	for (unsigned int k = 0; k < dim; k++) {

		  xi = fabs(x_i(k) - x_j(k))*theta(k);

		  double xi2 = xi*xi;   double xi3 = xi2*xi;   double xi4 = xi3*xi;

		  if (xi <= 0.4)
			  {ss(k) = 1 - 15*xi2 + 35*xi3 - 195.0/8*xi4;}
		  else if (xi < 1)
			  {ss(k) = 5.0/3 - 20.0/3*xi + 10*xi2 - 20.0/3*xi3 + 5.0/3*xi4;}
		  else
			  {ss(k) = 0;}
        }

	return prod(ss);
}

/* implementation according to the Forrester book */
/* void GEKModel::computeCorrelationMatrixDot(vec theta) {

	unsigned int numberOfSamples = data.getNumberOfSamples();
	mat X = data.getInputMatrix();

	//vec theta = GEK_weights;
	int k = X.n_cols;

	mat Psi=zeros(numberOfSamples,numberOfSamples);
	mat PsiDot=zeros(numberOfSamples,numberOfSamples);


	mat Rfull;

	for(int row = -1; row < k; row++){

		if(row == -1){ /* first row

			for(unsigned int i=0; i<numberOfSamples;i++){
				for(unsigned int j=i+1;j<numberOfSamples;j++){

					Psi(i,j)= computeCorrelation(X.row(i), X.row(j), theta);

				}
			}

			Psi = Psi+ trans(Psi)+ eye(numberOfSamples,numberOfSamples);

			Rfull=Psi;


			PsiDot=zeros(numberOfSamples,numberOfSamples);
			for(int l=0;l<k; l++){


				for(unsigned int i=0; i<numberOfSamples;i++){
					for(unsigned int j=0;j<numberOfSamples;j++){
						PsiDot(i,j)=2.0*theta(l)* (X(i,l)-X(j,l))*Psi(i,j);

					}
				}
				Rfull = join_rows(Rfull,PsiDot);

			}

		}

		else{ /* other rows

			mat Rrow;

			PsiDot=zeros(numberOfSamples,numberOfSamples);

			for(unsigned int i=0; i<numberOfSamples;i++){
				for(unsigned int j=0;j<numberOfSamples;j++){

					PsiDot(i,j)=-2.0*theta(row)* (X(i,row)-X(j,row))*Psi(i,j);

				}
			}

			Rrow = PsiDot;

			for(int l=0; l<k;l++){
				mat PsiDot2=zeros(numberOfSamples,numberOfSamples);

				if(l == row){
					for(unsigned int i=0; i<numberOfSamples;i++){
						for(unsigned int j=0;j<numberOfSamples;j++){
							PsiDot2(i,j)=
									(2.0*theta(l)-4.0*theta(l)*theta(l)* pow((X(i,l)-X(j,l)),2.0))*Psi(i,j);

						}
					}

				}

				else{


					for(unsigned int i=0; i<numberOfSamples;i++){
						for(unsigned int j=0;j<numberOfSamples;j++){

							PsiDot2(i,j)=
									(-4.0*theta(row)*theta(l)*(X(i,row)-X(j,row))*(X(i,l)-X(j,l)))*Psi(i,j);

						}
					}
				}

				Rrow = join_rows(Rrow,PsiDot2);
			}

			Rfull = join_cols(Rfull,Rrow);
		}

	} /* end of for loop for rows



	correlationMatrixDot  = Rfull + epsilonGEK * eye(numberOfSamples*(k+1),numberOfSamples*(k+1));



} */ /* end of compute_R_matrix_GEK */


vec GEKModel::computeCorrelationVectorDot(rowvec x) const{

	unsigned int dim             = data.getDimension();
	unsigned int numberOfSamples = data.getNumberOfSamples();

	mat X     = data.getInputMatrix();
	vec theta = GEK_weights;

	vec r(numberOfSamples*(dim+1));

	int count = 0;

	for(unsigned int i=0;i<numberOfSamples;i++){

		r(count) = computeCorrelation(x, X.row(i), theta);
		count++;
	}

	for(unsigned int i=0;i<dim;i++){
		for(unsigned int j=0;j<numberOfSamples;j++){

			r(count) = computedR_dxj(x, X.row(j),i);
			count++;

		}
	}

	return r;

}


void GEKModel::updateAuxilliaryFields(void){

	unsigned int dim = data.getDimension();
	unsigned int N = data.getNumberOfSamples();

#if 0
	cout<<"Updating auxiliary variables of the GEK model\n";
#endif

	vec pd                = zeros(N*dim) ;
	vec y                 = data.getOutputVector();
	mat gradientData      = data.getGradientMatrix();
	Bounds boxConstraints = data.getBoxConstraints();
	double std_y =  data.getOutputStd();

	yGEK.set_size(N*(dim+1));

	for(unsigned int i = 0; i< dim; i++){

		vec gradx = gradientData.col(i);

		for(unsigned int j=0; j<N; j++){

			double xmin = boxConstraints.getLowerBound(i);
			double xmax = boxConstraints.getUpperBound(i);
			//yGEK(numberOfSamples+i*numberOfSamples+j) = gradx(j)*( xmax - xmin )*dim;
			pd(i*N+j) = gradx(j)*( xmax - xmin )*dim/std_y;

		}
	}

	yGEK      = join_cols(y,pd);
	vec theta = GEK_weights;
	mat X     = data.getInputMatrix();

	correlationMatrixDot = correlationfunction.corrbiquadspline_gekriging(X,theta);

#if 0

	cout<<"\nCorrelation matrix\n";
	for(unsigned int i=0; i< correlationMatrixDot.n_cols; i++ ){

		vec Rdotcol = correlationMatrixDot.col(i);
		cout<<"Column: "<<i<<"\n";
		printVector(Rdotcol);


	}

#endif

	/* Cholesky decomposition R = LDL^T */

	correlationMatrixDot.save("correlationMatrix.csv",csv_ascii);

	int cholesky_return = chol(upperDiagonalMatrixDot, correlationMatrixDot);

	if (cholesky_return == 0) {

		printf("ERROR: Ill conditioned correlation matrix, Cholesky decomposition failed at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);
	}


#if 0
	printMatrix(upperDiagonalMatrixDot,"upperDiagonalMatrixDot");
#endif

	vec R_inv_ys(N*(dim+1)); R_inv_ys.fill(0.0);

	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_ys, yGEK);         /* solve R x = ys */

	R_inv_F = zeros(N*(dim+1));

	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_F, vectorOfF);      /* solve R x = F */

	beta0 = (1.0/dot(vectorOfF,R_inv_F)) * (dot(vectorOfF,R_inv_ys));

	vec ys_min_betaF = yGEK - beta0*vectorOfF;

	/* solve R x = ys-beta0*I */

	solveLinearSystemCholesky(upperDiagonalMatrixDot, R_inv_ys_min_beta , ys_min_betaF);

	sigmaSquared = (1.0 / (N*(dim+1))) * dot(ys_min_betaF, R_inv_ys_min_beta);

	// cout << "sigmaSquared is " << sigmaSquared << endl;
	// cout << "beta0 is " << beta0 << endl;
}

void GEKModel::addNewSampleToData(rowvec newsample){


	/* avoid points that are too close to each other */

	mat rawData = data.getRawData();

	//bool flagTooClose= checkifTooCLose(newsample, rawData);


	//if(!flagTooClose){

	appendRowVectorToCSVData(newsample, filenameDataInput);

	updateModelWithNewData();

	//}
	//else{

	//	std::cout<<"WARNING: The new sample is too close to a sample in the training data, it is discarded!\n";

	//}

}

void GEKModel::updateModelWithNewData(void){

	resetDataObjects();
	readData();

	unsigned int N   = data.getNumberOfSamples();
	unsigned int dim = data.getDimension();

	normalizeData();

	correlationMatrixDot.set_size(N*(dim+1),N*(dim+1));
	correlationMatrixDot.fill(0.0);
	upperDiagonalMatrixDot.set_size(N*(dim+1),N*(dim+1));
	upperDiagonalMatrixDot.fill(0.0);
	R_inv_ys_min_beta.set_size(N*(dim+1));
	R_inv_ys_min_beta.fill(0.0);
	R_inv_F.set_size(N*(dim+1));
	R_inv_F.fill(0.0);
	vectorOfF.set_size(N*(dim+1));
	vectorOfF.fill(1.0);

	updateAuxilliaryFields();

}

void GEKModel::resetDataObjects(void){

	correlationMatrixDot.reset();
	upperDiagonalMatrixDot.reset();
	R_inv_F.reset();
	R_inv_ys_min_beta.reset();
	vectorOfF.reset();
	beta0 = 0.0;
	sigmaSquared = 0.0;

}

void GEKModel::calculateExpectedImprovement(CDesignExpectedImprovement &currentDesign) const{

	double ftilde = 0.0;
	double ssqr   = 0.0;

	interpolateWithVariance(currentDesign.dv,&ftilde,&ssqr);

	#if 0
		printf("ftilde = %15.10f, ssqr = %15.10f\n",ftilde,ssqr);
	#endif

	double	sigma = sqrt(ssqr)	;

	#if 0
		printf("standart_ERROR = %15.10f\n",sigma);
	#endif

	double expectedImprovementValue = 0.0;

	if(fabs(sigma) > EPSILON){

		double ymin = data.getMinimumOutputVector();
		double	Z = (ymin - ftilde)/sigma;
	#if 0
		printf("Z = %15.10f\n",Z);
		printf("ymin = %15.10f\n",ymin);
	#endif


		expectedImprovementValue = (ymin - ftilde)*cdf(Z,0.0,1.0)+ sigma * pdf(Z,0.0,1.0);
	}
	else{

		expectedImprovementValue = 0.0;

		}
	#if 0
		printf("expectedImprovementValue = %20.20f\n",expectedImprovementValue);
	#endif

		currentDesign.objectiveFunctionValue = ftilde;
		currentDesign.valueExpectedImprovement = expectedImprovementValue;

}



/*void GEKModel::calculateExpectedImprovement_Grad(CDesignExpectedImprovement &currentDesign) const{  // Created by Kai

	double ftilde = 0.0;
	double ssqr   = 0.0;

	interpolateWithVariance(currentDesign.dv,&ftilde,&ssqr);

	double	sigma = sqrt(ssqr);

	double expectedImprovementValue = 0.0;

	if(fabs(sigma) > EPSILON){

		double ymin = data.getMinimumOutputVector();
		double	Z = (ymin - ftilde)/sigma;

	#if 0
		printf("Z = %15.10f\n",Z);
		printf("ymin = %15.10f\n",ymin);
	#endif


		expectedImprovementValue = (ymin - ftilde)*cdf(Z,0.0,1.0)+ sigma * pdf(Z,0.0,1.0);
	}
	else{

		expectedImprovementValue = 0.0;

		}
	#if 0
		printf("expectedImprovementValue = %20.20f\n",expectedImprovementValue);
	#endif

		currentDesign.objectiveFunctionValue = ftilde;
		currentDesign.valueExpectedImprovement = expectedImprovementValue;

} */

void GEKModel::boxmin(vec hyper_l, vec hyper_u, int num){

	dim = getDimension();

	mat hyper        = ones(dim,num);
	vec likeli_value = ones(num);

	vec log_ub = log10(hyper_u);
	vec log_lb = log10(hyper_l);
    vec random;  random.randu(num);

    hyper_cur      = hyper;
    likelihood_cur = likeli_value;

	for (unsigned int i=0;i<dim;i++){
	  for (unsigned int j=0;j<num;j++){
		  hyper(i,j) = pow(10,random(j)*(log_ub(i)-log_lb(i))+log_lb(i));
	  }
    }

	hyper_lb  = hyper_l;    //  lower bound
	hyper_up  = hyper_u;    //  upper bound


	//#pragma omp parallel for

	for (unsigned int kk=0;kk<num;kk++){      // Multi-starts

	  start(hyper.col(kk),hyper_lb,hyper_up,kk);

	  int kmax;

	  if (dim < 2)
	      { kmax = 2;}
      else
	      { kmax = std::min(dim,5);}

	  for (unsigned int k = 0; k < kmax; k++){  // Iterate for kmax times

	   vec hyper1 = hyper_cur.col(kk);

	   explore(hyper_cur.col(kk),likelihood_cur(kk),kk);

	   move(hyper1,hyper_cur.col(kk),likelihood_cur(kk),kk);

	 }

	}

	likeli_value = getLikelihood();
	hyper        = getTheta();

	uword i = likeli_value.index_min();

	likelihood_optimal = likeli_value(i);
	hyper_optimal      = hyper.col(i);

}

void GEKModel::start(vec hyper_in, vec hyper_l, vec hyper_u, int kk){

	  vec m = linspace(1,dim,dim)/(dim+2);
	  increment = zeros(dim);

	  for (unsigned int k = 0; k < dim; k++){
		  increment(k) = pow(2,m(k));
	   }

	  ind_increment = find(increment != 1);

	  hyper_cur.col(kk) = hyper_in;

	  likelihood_cur(kk) = likelihood_function(hyper_cur.col(kk));  //

	  numberOfIteration = 0;
      hyperoptimizationHistory = zeros(dim+2,200*dim);

	  hyperoptimizationHistory.col(numberOfIteration) = join_cols(hyper_cur.col(kk), vec { likelihood_cur(kk), 1.0} );


}

void GEKModel::explore(vec hyper_1, double likelihood_1, int kk){

	unsigned int j; double DD;  unsigned int atbd;

    hyper_cur.col(kk) = hyper_1; likelihood_cur(kk) = likelihood_1;

	for (unsigned int k = 0; k < size(ind_increment,0); k++){

	   j = ind_increment(k);
	   hyper_par = hyper_cur.col(kk);
       DD = increment(j);

       if (hyper_cur(j,kk) == hyper_up(j)){

    	   atbd = 1;
    	   hyper_par(j) =  hyper_cur(j,kk)/sqrt(DD); }

       else if (hyper_cur(j,kk) == hyper_lb(j)){

    	   atbd = 1;
    	   hyper_par(j) =  hyper_cur(j,kk)*sqrt(DD); }

       else  {

    	   atbd = 0;
    	   hyper_par(j) = std::min(hyper_up(j),hyper_cur(j,kk)*DD);
       }

       likelihood = likelihood_function(hyper_par);
       numberOfIteration++;
       hyperoptimizationHistory.col(numberOfIteration)= join_cols(hyper_par, vec { likelihood, 2});

       if (likelihood < likelihood_cur(kk)){
            likelihood_cur(kk) = likelihood;
            hyper_cur.col(kk) = hyper_par;  }
       else  {

    	    hyperoptimizationHistory(dim+1,numberOfIteration)= -2;

    	   if (!atbd) {
    		    hyper_par(j) = std::max(hyper_lb(j),hyper_cur(j,kk)/DD);
    	        likelihood = likelihood_function(hyper_par);

    	        numberOfIteration++;
    	        hyperoptimizationHistory.col(numberOfIteration)=join_cols(hyper_par, vec { likelihood, 2});

    	        if (likelihood < likelihood_cur(kk)){
    	        	likelihood_cur(kk) = likelihood;
    	        	hyper_cur.col(kk) = hyper_par;  }
    	        else
    	            hyperoptimizationHistory(dim+1,numberOfIteration)= -2;
    	    }
	     }
	  }

}

void GEKModel::move(vec hyper_old,vec hyper_new, double likelihood_new, int kk){

	   vec v  = hyper_new/hyper_old;
	   vec v1 = v-ones(dim,1);

       if (v1.is_zero()){

    	   vec ind = linspace(1,dim,dim);
    	   ind(dim-1) = 0;

    	   for (unsigned int k = 0; k < dim; k++){
    	  	    increment(k) = pow(increment(ind(k)),0.2);
    	    }

    	   likelihood_cur(kk) = likelihood_new;
    	   hyper_cur.col(kk) = hyper_new;

            return ;
        }

        unsigned int rept = 1;   likelihood_cur(kk) = likelihood_new;  hyper_cur.col(kk) = hyper_new;

        while (rept){

		   hyper_par = min(join_rows(hyper_up,max(join_rows(hyper_lb,hyper_new % v),1)),1);
		   likelihood = likelihood_function(hyper_par);
		   numberOfIteration++;
		   hyperoptimizationHistory.col(numberOfIteration)= join_cols(hyper_par, vec { likelihood, 3});

		   if (likelihood < likelihood_cur(kk)){
			   hyper_cur.col(kk) = hyper_par;
			   likelihood_cur(kk) = likelihood;
			   v = v % v;
		   }

		   else {
			   hyperoptimizationHistory(dim+1,numberOfIteration)= -3;
			   rept = 0;

		   }

		   if (size(find(hyper_par - hyper_up),0)+size(find(hyper_par - hyper_up),0) < 2*dim)
			    rept  =  0;
        }

         vec ind = linspace(1,dim,dim);
         ind(dim-1) = 0;

         for (unsigned int k = 0; k < dim; k++){
          	 increment(k) = pow(increment(ind(k)),0.25);
          }
}

mat GEKModel::getTheta(void) const{
	return hyper_cur;
}

vec GEKModel::getOptimalTheta(void) const{
	return hyper_optimal;
}


vec GEKModel::getLikelihood(void) const{
	return likelihood_cur;
}

double GEKModel::getOptimalLikelihood(void) const{
	return likelihood_optimal;
}


