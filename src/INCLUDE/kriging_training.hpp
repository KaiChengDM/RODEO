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

#ifndef TRAIN_KRIGING_HPP
#define TRAIN_KRIGING_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"
#include "surrogate_model.hpp"
#include "linear_regression.hpp"
#include "design.hpp"
#include "correlation_functions.hpp"


using namespace arma;

class KrigingModel : public SurrogateModel{

private:

	vec Kriging_weights;
	vec R_inv_ys_min_beta;
	vec R_inv_I;
	vec vectorOfOnes;
	mat correlationMatrix;
	mat upperDiagonalMatrix;

	std::vector<vec> Kriging_weights_vec;
	std::vector<vec> R_inv_ys_min_beta_vec;
	std::vector<mat> upperDiagonalMatrix_vec;
	std::vector<mat> correlationMatrix_vec;

	std::vector<double> likelihood_optimal_vec;
	std::vector<double> beta0_vec;
	std::vector<double> sigmaSquared_vec;


	double beta0 = 0.0;
	double sigmaSquared = 0.0;

	bool ifUsesLinearRegression = false;
	double epsilonKriging = 10E-12;

	double genErrorKriging;

	LinearModel linearModel;

	Correlationfunction correlationfunction;

	// void updateWithNewData(void);
	// void updateModelParams(void);

	void computeCorrelationMatrix(void);
	vec computeCorrelationVector(rowvec x, vec theta) const;
	double computeCorrelation(rowvec x_i, rowvec x_j, vec theta) const;
	double likelihood_function(vec theta);       // Modified by Kai

	/* Hooke Jeeves algorithm parameter*/

	 double likelihood;
	 int num;
     int rank;

     vec hyper_lb;
     vec hyper_up;
	 vec hyper_in;

	 mat hyper_cur;
	 vec hyper_par;
	 vec hyper_optimal;

	 vec increment;
	 uvec ind_increment;

	 mat hyperoptimizationHistory;

	 unsigned int numberOfIteration;
	 int dim;

	 vec likelihood_cur;
	 double likelihood_optimal;

public:

	KrigingModel();
	KrigingModel(std::string name);

	void setNameOfInputFile(std::string filename);
	void setNameOfHyperParametersFile(std::string filename);
	void setNumberOfTrainingIterations(unsigned int);

	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);
	void train(void);
	double interpolateWithGradients(rowvec x) const ;
	double interpolate(rowvec x) const ;
	vec    interpolate_vec(rowvec x) const ;

	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;
	void interpolateWithVariance_vec(rowvec xp,vec &f_tilde, vec &ssqr) const;


	void calculateExpectedImprovement(CDesignExpectedImprovement &currentDesign) const;
	void addNewSampleToData(rowvec newsample);

	double getyMin(void) const;

	vec getRegressionWeights(void) const;
	void setRegressionWeights(vec weights);
	void setEpsilon(double inp);
	void setLinearRegressionOn(void);
	void setLinearRegressionOff(void);

	void setTheta(vec theta);

	void resetDataObjects(void);
	void resizeDataObjects(void);


	void updateModelWithNewData(void);
	void updateAuxilliaryFields(void);

	/* Hooke Jeeves algorithm*/

	void boxmin(vec hyper_lb, vec hyper_ub, int num);
	void start(vec int_hyper, vec hyper_lb, vec hyper_ub, int num);
	void explore(vec int_hyper, double likelihood, int num);
	void move(vec hyper_1, vec hyper_2, double likelihood, int num);

	mat getTheta(void) const;
    vec getLikelihood(void) const;
    vec getOptimalTheta(void) const;
    double getOptimalLikelihood(void) const;

};



/* class EAdesign {
public:
	double fitness;             
	double objective_val;       
	vec theta;
	vec gamma;	
	double crossover_probability;
	double death_probability;
	//	double log_regularization_parameter;
	unsigned int id;

	void print(void);
	EAdesign(int dimension);
	int calculate_fitness(double epsilon, mat &X,vec &ys);

} ;



int calculate_fitness(EAdesign &new_born,
		double &reg_param,
		mat &R,
		mat &U,
		mat &L,
		mat &X,
		vec &ys,
		vec &I);



void pickup_random_pair(std::vector<EAdesign> population, int &mother,int &father);
void crossover_kriging(EAdesign &father, EAdesign &mother, EAdesign &child);
void update_population_properties(std::vector<EAdesign> &population); */


//int train_kriging_response_surface(KrigingModel &model);



#endif
