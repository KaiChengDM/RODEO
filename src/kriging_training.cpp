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
#include<cassert>

#include "kriging_training.hpp"
#include "linear_regression.hpp"
#include "auxiliary_functions.hpp"
#include "random_functions.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "correlation_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;

/* global variables */

int total_number_of_function_evals;

double population_overall_max = -10E14;
int population_overall_max_tread_id = -1;


KrigingModel::KrigingModel():SurrogateModel(){}


KrigingModel::KrigingModel(std::string nameInput):SurrogateModel(nameInput),linearModel(nameInput){

	modelID = ORDINARY_KRIGING;

	setNameOfHyperParametersFile(nameInput);

}


void KrigingModel::setNameOfInputFile(std::string filename){

	assert(!filename.empty());
	filenameDataInput = filename;
	linearModel.setNameOfInputFile(filename);

}

void KrigingModel::setNumberOfTrainingIterations(unsigned int nIters){

	numberOfTrainingIterations = nIters;

}

void KrigingModel::setNameOfHyperParametersFile(std::string label){

	assert(isNotEmpty(label));

	string filename = label + "_kriging_hyperparemeters.csv";

	output.printMessage("Name of the hyperparameter file is set as: ", filename);

	hyperparameters_filename = filename;

}


void KrigingModel::initializeSurrogateModel(void){

	assert(ifDataIsRead);
	assert(ifNormalized);

	output.printMessage("Initializing the Kriging model...");

	unsigned int dim             = data.getDimension();
	unsigned int numberOfSamples = data.getNumberOfSamples() ;

	numberOfHyperParameters = dim;
	Kriging_weights         = ones<vec>(dim);
	correlationMatrix       = zeros<mat>(numberOfSamples,numberOfSamples);
	upperDiagonalMatrix     = zeros<mat>(numberOfSamples,numberOfSamples);
	R_inv_ys_min_beta       = zeros<vec>(numberOfSamples);
	R_inv_I                 = zeros<vec>(numberOfSamples);
	vectorOfOnes            = ones<vec>(numberOfSamples);

	if (ifVectorOutput){           // Check if the output is a vector

		rank = readRank();

	 for (unsigned int i=0; i< rank; i++){

		  Kriging_weights_vec.push_back(Kriging_weights);
		  R_inv_ys_min_beta_vec.push_back(R_inv_ys_min_beta);
		  upperDiagonalMatrix_vec.push_back(upperDiagonalMatrix);
	      correlationMatrix_vec.push_back(correlationMatrix);
	  }

	 }

	if(ifUsesLinearRegression){    // currently, we don't use linear regression

		output.printMessage("Linear model is active for the Kriging model...");

		if(areGradientsOn()) {
			linearModel.setGradientsOn();
		}

		linearModel.readData();
		linearModel.setBoxConstraints(data.getBoxConstraints());


		linearModel.normalizeData();
		linearModel.initializeSurrogateModel();
		linearModel.train();

	}

	updateAuxilliaryFields();

	ifInitialized = true;

#if 0
	std::cout << "Kriging model initialization is done...\n";
#endif

}

void KrigingModel::printHyperParameters(void) const{

	output.printMessage("Hyperparameters of the Kriging model = ");
	output.printMessage("theta",Kriging_weights);
	//output.printMessage("gamma",gamma);

}
void KrigingModel::saveHyperParameters(void) const{


	if(!hyperparameters_filename.empty()){

		output.printMessage("Saving hyperparameters into the file: ", hyperparameters_filename);

		unsigned int dim = data.getDimension();

		rowvec saveBuffer(numberOfHyperParameters);
		for(unsigned int i=0; i<dim; i++) saveBuffer(i) = Kriging_weights(i);
		for(unsigned int i=0; i<dim; i++) saveBuffer(i+dim) = gamma(i);

		saveBuffer.save(hyperparameters_filename,csv_ascii);

	}

}

void KrigingModel::loadHyperParameters(void){


	rowvec loadBuffer;
	bool ifLoadIsOK =  loadBuffer.load(hyperparameters_filename,csv_ascii);

	unsigned int numberOfEntriesInTheBuffer = loadBuffer.size();

	if(ifLoadIsOK && numberOfEntriesInTheBuffer == numberOfHyperParameters) {

		unsigned int dim = data.getDimension();

		for(unsigned int i=0; i<dim; i++) Kriging_weights(i) = loadBuffer(i);
		//for(unsigned int i=0; i<dim; i++) gamma(i) = loadBuffer(i+dim);


	}

}

double KrigingModel::getyMin(void) const{

	return min(data.getOutputVector());

}

/*vec KrigingModel::getTheta(void) const{

	return theta;


}

vec KrigingModel::getGamma(void) const{

	return gamma;

}

void KrigingModel::setGamma(vec gamma){

	this->gamma = gamma;
}

*/

void KrigingModel::setTheta(vec theta){

	this->Kriging_weights = theta;

}


vec KrigingModel::getRegressionWeights(void) const{

	return linearModel.getWeights();

}

void KrigingModel::setRegressionWeights(vec weights){

	linearModel.setWeights(weights);

}


void KrigingModel::setEpsilon(double value){

	assert(value>0);

	epsilonKriging = value;

}

void KrigingModel::setLinearRegressionOn(void){

	ifUsesLinearRegression  = true;
	modelID = UNIVERSAL_KRIGING;

}
void KrigingModel::setLinearRegressionOff(void){

	ifUsesLinearRegression  = false;
	modelID = ORDINARY_KRIGING;
}




/** Adds the rowvector newsample to the data of the Kriging model and updates model parameters
 * @param[in] newsample
 *
 */

void KrigingModel::addNewSampleToData(rowvec newsample){


	/* avoid points that are too close to each other */

	mat rawData = data.getRawData();

	bool flagTooClose= checkifTooCLose(newsample, rawData);


	if(!flagTooClose){

		appendRowVectorToCSVData(newsample, filenameDataInput);

		updateModelWithNewData();

	}
	else{

		std::cout<<"WARNING: The new sample is too close to a sample in the training data, it is discarded!\n";

	}

}


void KrigingModel::printSurrogateModel(void) const{
	cout << "\nKriging Surrogate model:\n";
	cout<< "Number of samples: "<<data.getNumberOfSamples()<<endl;
	cout<< "Number of input parameters: "<<data.getDimension() <<"\n";

#if 0
	printMatrix(rawData,"rawData");
	printMatrix(X,"X");
#endif

	printf("hyperparameters_filename: %s\n",hyperparameters_filename.c_str());
	printf("input_filename: %s\n",filenameDataInput.c_str());
	printf("max_number_of_kriging_iterations = %d\n",this->numberOfTrainingIterations);
	printf("epsilonKriging = %15.10e\n",epsilonKriging);
	printHyperParameters();

#if 0
	printVector(R_inv_ys_min_beta,"R_inv_ys_min_beta");
#endif
	std::cout<<"beta0 = "<<beta0<<"\n";
#if 0
	printMatrix(correlationMatrix,"correlationMatrix");
#endif
	printf("\n");

}


void KrigingModel::resetDataObjects(void){

	correlationMatrix.reset();
	upperDiagonalMatrix.reset();
	R_inv_I.reset();
	R_inv_ys_min_beta.reset();
	vectorOfOnes.reset();
	beta0 = 0.0;
	sigmaSquared = 0.0;

	std::vector<vec> tmp1;
	Kriging_weights_vec.swap(tmp1);
	std::vector<vec> tmp2;
	R_inv_ys_min_beta_vec.swap(tmp2);

	std::vector<double> tmp3;
	likelihood_optimal_vec.swap(tmp3);;
	std::vector<double> tmp4;
	beta0_vec.swap(tmp4);
	std::vector<double> tmp5;
    sigmaSquared_vec.swap(tmp5);

	std::vector<mat> tmp6;
    upperDiagonalMatrix_vec.swap(tmp6);
	std::vector<mat> tmp7;
	correlationMatrix_vec.swap(tmp7);

}

void KrigingModel::resizeDataObjects(void){

	unsigned int numberOfSamples = data.getNumberOfSamples();

	correlationMatrix.set_size(numberOfSamples,numberOfSamples);
	correlationMatrix.fill(0.0);
	upperDiagonalMatrix.set_size(numberOfSamples,numberOfSamples);
	upperDiagonalMatrix.fill(0.0);
	R_inv_ys_min_beta.set_size(numberOfSamples);
	R_inv_ys_min_beta.fill(0.0);
	R_inv_I.set_size(numberOfSamples);
	R_inv_I.fill(0.0);
	vectorOfOnes.set_size(numberOfSamples);
	vectorOfOnes.fill(1.0);

}



void KrigingModel::updateAuxilliaryFields(void){

	unsigned int numberOfSamples = data.getNumberOfSamples();

#if 0
	cout<<"Updating auxiliary variables of the Kriging model\n";
#endif

	mat X = data.getInputMatrix();

	if (ifVectorOutput){

		for (unsigned int i = 0; i< rank; i++){

		   assignOutput(i);               // assign the i-th POD coefficient to the output

		   vec ys = data.getOutputVector();

		   vec theta = Kriging_weights_vec[i];

		   //  	correlationMatrix_vec[i] = correlationfunction.corrbiquadspline_kriging(X,theta);

		     correlationMatrix_vec[i] = correlationfunction.corrgaussian_kriging(X,theta);

		   	/* Cholesky decomposition R = LDL^T */

		   	int cholesky_return = chol(upperDiagonalMatrix_vec[i], correlationMatrix_vec[i]);

		   	if (cholesky_return == 0) {

		   		printf("ERROR: Ill conditioned correlation matrix, Cholesky decomposition failed!\n");
		   		abort();
		   	}

		   	vec R_inv_ys(numberOfSamples);

		   	R_inv_ys.fill(0.0);

		   	solveLinearSystemCholesky(upperDiagonalMatrix_vec[i], R_inv_ys, ys);    /* solve R x = ys */

			R_inv_I = zeros(numberOfSamples);

		   	solveLinearSystemCholesky(upperDiagonalMatrix_vec[i], R_inv_I, vectorOfOnes);      /* solve R x = I */

		   	beta0 = (1.0/dot(vectorOfOnes,R_inv_I)) * (dot(vectorOfOnes,R_inv_ys));

		   	beta0_vec.push_back(beta0);

		   	vec ys_min_betaI = ys - beta0*vectorOfOnes;

		   	/* solve R x = ys-beta0*I */

		   	solveLinearSystemCholesky(upperDiagonalMatrix_vec[i], R_inv_ys_min_beta_vec[i] , ys_min_betaI);

		   	sigmaSquared = (1.0 / numberOfSamples) * dot(ys_min_betaI, R_inv_ys_min_beta_vec[i]);

		   	sigmaSquared_vec.push_back(sigmaSquared);

		}

	}else{

		vec ys = data.getOutputVector();

		if(ifUsesLinearRegression){

				vec ysLinearRegression = linearModel.interpolateAll(X);
				ys = ys - ysLinearRegression;

		}

			vec theta = Kriging_weights;

			// correlationMatrix = correlationfunction.corrbiquadspline_kriging(X,theta);

			correlationMatrix = correlationfunction.corrgaussian_kriging(X,theta);

			/* Cholesky decomposition R = LDL^T */

			int cholesky_return = chol(upperDiagonalMatrix, correlationMatrix);

			if (cholesky_return == 0) {
				printf("ERROR: Ill conditioned correlation matrix, Cholesky decomposition failed!\n");
				abort();
			}

			vec R_inv_ys(numberOfSamples);

			R_inv_ys.fill(0.0);

			solveLinearSystemCholesky(upperDiagonalMatrix, R_inv_ys, ys);    /* solve R x = ys */

			R_inv_I = zeros(numberOfSamples);

			solveLinearSystemCholesky(upperDiagonalMatrix, R_inv_I, vectorOfOnes);      /* solve R x = I */

			beta0 = (1.0/dot(vectorOfOnes,R_inv_I)) * (dot(vectorOfOnes,R_inv_ys));

			vec ys_min_betaI = ys - beta0*vectorOfOnes;

			/* solve R x = ys-beta0*I */

			solveLinearSystemCholesky(upperDiagonalMatrix, R_inv_ys_min_beta , ys_min_betaI);

			sigmaSquared = (1.0 / numberOfSamples) * dot(ys_min_betaI, R_inv_ys_min_beta);
	}

}

void KrigingModel::updateModelWithNewData(void){

	resetDataObjects();

	readData();

	unsigned int numberOfSamples = data.getNumberOfSamples();

	normalizeData();

	/*correlationMatrix.set_size(numberOfSamples,numberOfSamples);
	correlationMatrix.fill(0.0);
	upperDiagonalMatrix.set_size(numberOfSamples,numberOfSamples);
	upperDiagonalMatrix.fill(0.0);
	R_inv_ys_min_beta.set_size(numberOfSamples);
	R_inv_ys_min_beta.fill(0.0);
	R_inv_I.set_size(numberOfSamples);
	R_inv_I.fill(0.0);
	vectorOfOnes.set_size(numberOfSamples);
	vectorOfOnes.fill(1.0);*/

	initializeSurrogateModel();  // reset everything when a new data is added to model

	// updateAuxilliaryFields();

}

vec KrigingModel::computeCorrelationVector(rowvec x, vec theta) const{

	unsigned int numberOfSamples = data.getNumberOfSamples();

	vec r(numberOfSamples);
	mat X = data.getInputMatrix();

	for(unsigned int i=0;i<numberOfSamples;i++){

		r(i) = computeCorrelation(x, data.getRowX(i),theta);

	}

	return r;

}


double KrigingModel::interpolate(rowvec xp ) const{

	double estimateLinearRegression = 0.0;
	double estimateKriging = 0.0;

	if(ifUsesLinearRegression ){

		estimateLinearRegression = linearModel.interpolate(xp);
	}

	vec theta = Kriging_weights;

	vec r = computeCorrelationVector(xp,theta);

	estimateKriging = beta0 + dot(r,R_inv_ys_min_beta);

	return estimateLinearRegression + estimateKriging;

}

vec KrigingModel::interpolate_vec(rowvec xp) const{

  /* double estimateLinearRegression = 0.0;

	double estimateKriging = 0.0;

	if(ifUsesLinearRegression ){

		estimateLinearRegression = linearModel.interpolate(xp);
	} */

	vec mean; mean.zeros(rank);

	for (unsigned int i=0; i< rank; i++){

		vec theta = Kriging_weights_vec[i];

		vec r = computeCorrelationVector(xp,theta);

		mean(i) = beta0_vec[i] + dot(r,R_inv_ys_min_beta_vec[i]);

	}

	return mean;

}


void KrigingModel::calculateExpectedImprovement(CDesignExpectedImprovement &currentDesign) const{

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



void KrigingModel::interpolateWithVariance(rowvec xp,double *ftildeOutput,double *sSqrOutput) const{

	assert(this->ifInitialized);

	unsigned int N = data.getNumberOfSamples();

	*ftildeOutput =  interpolate(xp);

	vec R_inv_r(N);

	vec theta = Kriging_weights;

	vec r = computeCorrelationVector(xp,theta);

	solveLinearSystemCholesky(upperDiagonalMatrix, R_inv_r, r);  // solve the linear system R x = r by Cholesky matrices U and L

	*sSqrOutput = sigmaSquared*( 1.0 - dot(r,R_inv_r)+ ( pow( (dot(r,R_inv_I) -1.0 ),2.0)) / (dot(vectorOfOnes,R_inv_I) ) );

	//cout << "predict MSE is " << sigmaSquared << endl;

}

void KrigingModel::interpolateWithVariance_vec(rowvec xp,vec &ftildeOutput,vec & sSqrOutput) const{

	assert(this->ifInitialized);

	unsigned int N = data.getNumberOfSamples();

	vec R_inv_r(N);

	ftildeOutput.zeros(rank);  sSqrOutput.zeros(rank);

	for (unsigned int i=0; i< rank; i++){

		vec theta = Kriging_weights_vec[i];

		vec r = computeCorrelationVector(xp,theta);

		ftildeOutput(i) = beta0_vec[i] + dot(r,R_inv_ys_min_beta_vec[i]);

	    solveLinearSystemCholesky(upperDiagonalMatrix_vec[i], R_inv_r, r);  // solve the linear system R x = r by Cholesky matrices U and L

	    sSqrOutput(i) = sigmaSquared_vec[i]*( 1.0 - dot(r,R_inv_r)+ ( pow( (dot(r,R_inv_I) -1.0 ),2.0)) / (dot(vectorOfOnes,R_inv_I) ) );

	}

}


double KrigingModel::computeCorrelation(rowvec x_i, rowvec x_j,vec theta) const {

	unsigned int dim = data.getDimension();
	double sum = 0.0;

	for (unsigned int k = 0; k < dim; k++) {
	   //sum += theta(k) * pow(fabs(x_i(k) - x_j(k)), gamma(k));
		sum += theta(k) * pow(fabs(x_i(k) - x_j(k)), 2);
	}

	return exp(-sum);
}

/*
double KrigingModel::computeCorrelation(rowvec x_i, rowvec x_j,vec theta) const {

	unsigned int dim = data.getDimension();
	double xi = 0.0;
	vec ss(dim);
	ss.fill(1);

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
}
*/

void KrigingModel::computeCorrelationMatrix(void)  {

	unsigned int N = data.getNumberOfSamples();

	mat X = data.getInputMatrix();

    mat identityMatrix = eye(N,N);

	correlationMatrix.fill(0.0);

	vec theta = Kriging_weights;

	 for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = i + 1; j < N; j++) {
			double R = computeCorrelation(data.getRowX(i), data.getRowX(j),theta);
			correlationMatrix(i, j) = R;
			correlationMatrix(j, i) = R;
		}

	}

	correlationMatrix = correlationMatrix + identityMatrix + identityMatrix*epsilonKriging;

	//correlationMatrix = correlationfunction.corrgaussian_kriging(X, theta);
}


double KrigingModel::likelihood_function(vec theta){

	unsigned int dim = data.getDimension();
	unsigned int N   = data.getNumberOfSamples();
	int mn = N;

	mat X = data.getInputMatrix();

	vec y = data.getOutputVector();

	// correlationMatrix = correlationfunction.corrbiquadspline_kriging(X,theta);

	correlationMatrix = correlationfunction.corrgaussian_kriging(X,theta);   // using Guassian kernel

	upperDiagonalMatrix = chol(correlationMatrix);

	long double logdetR = 2*sum(log(diagvec(upperDiagonalMatrix)));

	vec R_inv_ys(mn); R_inv_ys.fill(0.0);

	solveLinearSystemCholesky(upperDiagonalMatrix, R_inv_ys, y);    /* solve R x = ys */

	R_inv_I = zeros(mn);

	solveLinearSystemCholesky(upperDiagonalMatrix, R_inv_I, vectorOfOnes);      /* solve R x = F */

	beta0 = (1.0/dot(vectorOfOnes,R_inv_I)) * (dot(vectorOfOnes,R_inv_ys));

	vec ys_min_betaF = y - beta0*vectorOfOnes;

	/* solve R x = ys-beta0*I */

	solveLinearSystemCholesky(upperDiagonalMatrix, R_inv_ys_min_beta , ys_min_betaF);

	sigmaSquared = (1.0 / (mn)) * dot(ys_min_betaF, R_inv_ys_min_beta);

    likelihood = mn * log(sigmaSquared) + logdetR;

    return likelihood;

}

void KrigingModel::train(void){


	if(!ifInitialized){
		initializeSurrogateModel();
	}

	unsigned int dim = data.getDimension();

	vec hyper_l = 0.0001*dim*ones(dim,1);
	vec hyper_u = 10*dim*ones(dim,1);

	num = 10;   // Multiple starts

	if (ifVectorOutput){

	   for (unsigned int i = 0; i< rank; i++){

		  assignOutput(i);

		  boxmin(hyper_l,hyper_u,num);    // Hooke Jeeves algorithm for hyper-parameter optimization

		  Kriging_weights_vec[i] = getOptimalTheta();

		  likelihood_optimal_vec.push_back(getOptimalLikelihood());

	   }

	  }else {

		  boxmin(hyper_l,hyper_u,num);

		  Kriging_weights = getOptimalTheta();

		  likelihood_optimal  = getOptimalLikelihood();
	  }

    //cout << "Optimal likelihood is " << likelihood_optimal << endl;

	updateAuxilliaryFields();

}


void KrigingModel::boxmin(vec hyper_l, vec hyper_u, int num){

	dim = getDimension();

	mat hyper = ones(dim,num);
	vec likeli_value = ones(num);

	vec log_ub = log10(hyper_u);
	vec log_lb = log10(hyper_l);
    vec random;  random.randu(num);

    hyper_cur = hyper;
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
	      { kmax = std::min(dim,4);}

	  for (unsigned int k = 0; k < kmax; k++){  // Iterate for kmax times

	   vec hyper1 = hyper_cur.col(kk);

	   explore(hyper_cur.col(kk),likelihood_cur(kk),kk);

	   move(hyper1,hyper_cur.col(kk),likelihood_cur(kk),kk);

	   // cout << "current likelihood is " << likelihood_cur(kk) << endl;

	 }

	}

	likeli_value = getLikelihood();
	hyper = getTheta();

	uword i = likeli_value.index_min();

	likelihood_optimal = likeli_value(i);
	hyper_optimal = hyper.col(i);

}

void KrigingModel::start(vec hyper_in, vec hyper_l, vec hyper_u, int kk){

	  vec m = linspace(1,dim,dim)/(dim+2);
	  increment = zeros(dim);

	  for (unsigned int k = 0; k < dim; k++){
		  increment(k) = pow(2,m(k));
	   }

	  ind_increment = find(increment != 1);

	  hyper_cur.col(kk) = hyper_in;

	  likelihood_cur(kk) = likelihood_function(hyper_cur.col(kk));

	  numberOfIteration = 0;
      hyperoptimizationHistory = zeros(dim+2,200*dim);

	  hyperoptimizationHistory.col(numberOfIteration) = join_cols(hyper_cur.col(kk), vec { likelihood_cur(kk), 1.0} );


}

void KrigingModel::explore(vec hyper_1, double likelihood_1, int kk){

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

void KrigingModel::move(vec hyper_old,vec hyper_new, double likelihood_new, int kk){

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

mat KrigingModel::getTheta(void) const{
	return hyper_cur;
}

vec KrigingModel::getOptimalTheta(void) const{
	return hyper_optimal;
}


vec KrigingModel::getLikelihood(void) const{
	return likelihood_cur;
}

double KrigingModel::getOptimalLikelihood(void) const{
	return likelihood_optimal;
}

/*void KrigingModel::train(void){

	total_number_of_function_evals = 0;

	assert(ifInitialized);

	unsigned int dim = data.getDimension();

	output.printMessage("Training Kriging response surface for the data: " + filenameDataInput);

	vec y = data.getOutputVector();

	vec ysKriging = y;


	if(ifUsesLinearRegression ){

		output.printMessage("Linear regression is active...");

		linearModel.train();

		mat X = data.getInputMatrix();
		vec ysLinearModel = linearModel.interpolateAll(X);

		ysKriging = ysKriging - ysLinearModel;
	}
	else{

		output.printMessage("Linear regression is not active...");

	}

	int max_number_of_function_calculations =  this->numberOfTrainingIterations;

	// initialize random seed
	srand (time(NULL));

	// get the number of treads
	int number_of_treads = 1;
#pragma omp parallel
	{

		int tid = omp_get_thread_num();
		number_of_treads = omp_get_num_threads();


		if (tid == 0){

			max_number_of_function_calculations = max_number_of_function_calculations/number_of_treads;

			output.printMessage("number of threads used :",number_of_treads);
			output.printMessage("number of function evaluations per thread :",max_number_of_function_calculations);

		}

	}


	int number_of_initial_population = dim * 16 / number_of_treads;

	if(number_of_initial_population < 100) {

		number_of_initial_population = 100;
	}


#pragma omp parallel
	{

		int tid;

		std::vector<EAdesign>::iterator it;
		std::vector<EAdesign> population;
		int population_size = 0;

		double population_max = -LARGE;
		int population_max_index = -1;


		mat Xmatrix = data.getInputMatrix();


		vec Ivector = vectorOfOnes;
		vec ys = ysKriging;

		unsigned int d = dim;
		double epsilon = epsilonKriging;


		double Jnormalization_factor = 0.0;



		EAdesign initial_design(d);

		initial_design.theta = generateRandomVector(0.0, 10.0, d);
		initial_design.gamma = generateRandomVector(0.0, 2.0, d);

		if(file_exist(hyperparameters_filename.c_str())){
#pragma omp master
			{
				output.printMessage("Found hyperparameter file: " + hyperparameters_filename);

			}


			loadHyperParameters();

			for(unsigned int i=0; i<d;i++) {

				if(theta(i) < 100.0){

					initial_design.theta(i) = theta(i);
				}

			}
			for(unsigned int i=0; i<d;i++) {

				if( gamma(i) < 2.0) {

					initial_design.gamma(i) = gamma(i);

				}
			}


		}
#if 1
#pragma omp master
		{

		}
#endif

		initial_design.calculate_fitness(epsilon,Xmatrix,ys);


		if (initial_design.objective_val != -LARGE) {

			Jnormalization_factor = initial_design.objective_val;
			initial_design.objective_val = 0.0;
			initial_design.id = population_size;
			population.push_back(initial_design);
			population_size++;

		}

		for (int i = 0; i < max_number_of_function_calculations; i++) {

			// update population properties after initital iterations
			if (i >= number_of_initial_population) {

				update_population_properties (population);
			}


			//			if(this->ifDisplay){
			//
			//				if (total_number_of_function_evals % 100 == 0){
			//
			//					printf("\r# of function calculations =  %d\n",
			//							total_number_of_function_evals);
			//					fflush(stdout);
			//
			//				}
			//
			//
			//
			//
			//			}


			EAdesign new_born(d);


			if (i < number_of_initial_population) {

				for (unsigned int j = 0; j < d; j++) {

					new_born.theta(j) = generateRandomDouble(0, 10); //theta
					new_born.gamma(j) = generateRandomDouble(0, 2); //gamma

				}

			} else {

				int father_id = -1;
				int mother_id = -1;
				pickup_random_pair(population, mother_id, father_id);

				crossover_kriging(population[mother_id], population[father_id],new_born);
			}

#if 0
			new_born.theta.print();
			new_born.gamma.print();
#endif

			mat X = data.getInputMatrix();
			new_born.calculate_fitness(epsilon,X,ys);


			if (new_born.objective_val != -LARGE) {

				if(Jnormalization_factor != 0){

					new_born.objective_val = new_born.objective_val-Jnormalization_factor;

				}
				else{
					Jnormalization_factor = new_born.objective_val;
					new_born.objective_val = 0.0;


				}



				new_born.id = population_size;

				population.push_back(new_born);
				population_size++;


			}

			if (population_size % 10 == 0 && i > number_of_initial_population) { //in each 10 iteration find population max

				population_max = -LARGE;
				for (it = population.begin(); it != population.end(); ++it) {

					if (it->objective_val > population_max) {
						population_max = it->objective_val;
						population_max_index = it->id;
					}

				}


			}

		} // end of for loop


		population_max= -LARGE;
		for (it = population.begin() ; it != population.end(); ++it){

			if ( it->objective_val >  population_max){
				population_max = it->objective_val;
				population_max_index = it->id;
			}


		}


#pragma omp barrier

#if 0
		printf("kriging model training is over...\n");
		printf("tread %d best design log likelihood = %10.7f\n",omp_get_thread_num(),population_max );
#endif



#pragma omp critical
		{
			if (population_max > population_overall_max){
#if 0
				printf("tread %d, population_overall_max = %10.7f\n",omp_get_thread_num(),population_overall_max);
#endif
				population_overall_max = population_max;
				population_overall_max_tread_id = omp_get_thread_num();
			}
		}


#pragma omp barrier


		tid = omp_get_thread_num();
		if(tid == population_overall_max_tread_id){


#if 0
			printf("tread %d has the best design with the log likelihood = %10.7f\n",population_overall_max_tread_id,population_overall_max );

			population.at(population_max_index).print();
#endif


			theta = population.at(population_max_index).theta;
			gamma = population.at(population_max_index).gamma;

			saveHyperParameters();

		}

	} // end of parallel section


	output.printMessage("Kring training is done...");

	printHyperParameters();


	updateAuxilliaryFields();

}*/



/*

EAdesign::EAdesign(int dimension){

	theta = zeros(dimension);
	gamma = zeros(dimension);


}

void EAdesign::print(void){

	printf("id = %d,  J = %10.7f, fitness =  %10.7f cross-over p = %10.7f\n",id,objective_val,fitness,crossover_probability);
#if 0
	printf("theta = \n");
	theta.print();
	if(gamma.size() > 0){
		printf("gamma = \n");
		gamma.print();
	}
#endif
}


int EAdesign::calculate_fitness(double epsilon, mat &X,vec &ys){

	double beta0 = 0.0;
	double ssqr = 0.0;
	double logdetR = 0.0;
	double objVal = 0.0;

	unsigned int N = X.n_rows;  // number of samples


	mat R(N,N,fill::zeros);
	mat U(N,N,fill::zeros);

	vec I(N,fill::ones);

	// compute the correlation matrix R

	//	double reg_param = pow(10.0,-1.0*new_born.log_regularization_parameter);

	compute_R_matrix(theta, gamma,epsilon,R,X);

#if 0
	R.raw_print(cout, "R:");
#endif


	// compute Cholesky decomposition
	int flag = chol(U, R);

	if (flag == 0) {

		printf("Ill conditioned correlation matrix! \n");
		objective_val = -LARGE;
		total_number_of_function_evals++;
		return 0;
	}



	vec U_diagonal = U.diag();
#if 0
	U_diagonal.raw_print(cout, "diag(U):");
#endif
	logdetR = 0.0;

	for (unsigned int i = 0; i < N; i++) {

		if (U_diagonal(i) < 0) {

			objective_val = -LARGE;
			total_number_of_function_evals++;
			return 0;
		}

		logdetR += log(U_diagonal(i));


	}

	logdetR = 2.0 * logdetR;

	vec R_inv_ys(N,fill::zeros);
	vec R_inv_I(N,fill::zeros);

	solveLinearSystemCholesky(U, R_inv_ys, ys); // solve R x = ys
	solveLinearSystemCholesky(U, R_inv_I, I);   // solve R x = I



	beta0 = (1.0/dot(I,R_inv_I)) * (dot(I,R_inv_ys));
#if 0
	printf("beta0= %20.15f\n",beta0);
#endif


	vec ys_min_betaI = ys-beta0*I;

	vec R_inv_ys_min_beta(N,fill::zeros);


	// solve R x = ys-beta0*I

	solveLinearSystemCholesky(U, R_inv_ys_min_beta, ys_min_betaI);


	double fac = dot(ys_min_betaI, R_inv_ys_min_beta);

	double oneOverN = 1.0/double(N);
	double NoverTwo = double(N)/2.0;

	ssqr = oneOverN * fac;


	if(ssqr > 0 ){
		objVal = (- NoverTwo) * log(ssqr);
		objVal -= 0.5 * logdetR;
	}
	else{

		objective_val = -LARGE;
		total_number_of_function_evals++;
		return 0;
	}

#if 0
	printf("\n");
	printf("objective function value = %10.7f\n",objective_val);
	printf("s^2= %20.15f\n",ssqr);
	printf("(-N/2.0)* log(ssqr) = %10.7f\n",(-NoverTwo)* log(ssqr));
	printf("log(ssqr) = %10.7f\n",log(ssqr));
	printf("-0.5*logdetR = %10.7f\n",-0.5*logdetR);
	printf("\n");
#endif


	total_number_of_function_evals++;

	objective_val = objVal;

	return 0;



}*/

/*

// print all the population (can be a lot of mess!)

void print_population(std::vector<EAdesign> population){

	std::vector<EAdesign>::iterator it;

	printf("\nPopulation:\n");

	for (it = population.begin() ; it != population.end(); ++it){

		it->print();

	}
	printf("\n");

}


// crossover function of two designs
void crossover_kriging(EAdesign &father, EAdesign &mother, EAdesign &child) {

	int dim = father.theta.size();
	for (int i = 0; i < dim; i++) {
		//		printf("theta (m) = %10.7f theta (f) = %10.7f\n",mother.theta[i],father.theta[i]);
		//		printf("gamma (m) = %10.7f gamma (f) = %10.7f\n",mother.gamma[i],father.gamma[i]);
		child.theta(i) = generateRandomDoubleFromNormalDist(father.theta(i), mother.theta(i), 4.0);
		child.gamma(i) = generateRandomDoubleFromNormalDist(father.gamma(i), mother.gamma(i), 4.0);

		if (child.theta(i) < 0){
			child.theta(i) = generateRandomDouble(0.0, 10.0);
		}

		if (child.gamma(i) < 0){
			child.gamma(i) = generateRandomDouble(0.1, 2.0);
		}

		if (child.gamma(i) > 2.0){
			child.gamma(i) = generateRandomDouble(0.1, 2.0);
		}

#if 0
		if (i==0) printf("theta = %10.7f gamma = %10.7f\n",child.theta[i],child.gamma[i]);
#endif
	}


	//child.log_regularization_parameter =
	//		random_number(father.log_regularization_parameter,
	//				mother.log_regularization_parameter, 4.0);
    //

	//if(child.log_regularization_parameter < 0 ) child.log_regularization_parameter=0.0;
	//if(child.log_regularization_parameter > 14 ) child.log_regularization_parameter=14.0;




}



void pickup_random_pair(std::vector<EAdesign> population, int &mother,int &father) {

	double rand_num;
	std::vector<EAdesign>::iterator it;

	//usleep(1000000);

	rand_num = rand() % 1000000;
	rand_num = rand_num / 1000000;

	double pro_sum = 0.0;
	for (it = population.begin(); it != population.end(); ++it) {

		pro_sum = pro_sum + it->crossover_probability;

		if (rand_num < pro_sum) {

			mother = it->id;
			break;
		}
	}

	while (1) {

		rand_num = rand() % 1000000;
		rand_num = rand_num / 1000000;

		double pro_sum = 0.0;
		for (it = population.begin(); it != population.end(); ++it) {
			pro_sum = pro_sum + it->crossover_probability;

			if (rand_num < pro_sum) {

				father = it->id;
				break;
			}
		}

		if (mother != father)
			break;

	}

}

void update_population_properties(std::vector<EAdesign> &population) {
	std::vector<EAdesign>::iterator it;
	double sum_fitness;
	double max, min;
	double x_std;
	sum_fitness = 0.0;
	max = -LARGE;
	min = LARGE;

	for (it = population.begin(); it != population.end(); ++it) {

		if (it->objective_val < min)
			min = it->objective_val;
		if (it->objective_val > max)
			max = it->objective_val;

	}


	for (it = population.begin(); it != population.end(); ++it) {

		x_std = (it->objective_val - min) / (max - min);
		it->fitness = x_std * 100.0;

		sum_fitness = sum_fitness + it->fitness;

	}

	for (it = population.begin(); it != population.end(); ++it) {


		it->crossover_probability = it->fitness / sum_fitness;
	}

}*/








