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

#ifndef GEK_HPP
#define GEK_HPP
#include "Rodeo_macros.hpp"
#include "surrogate_model.hpp"
#include "linear_regression.hpp"
#include "correlation_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
using namespace arma;


class GEKModel : public SurrogateModel{

private:

	vec GEK_weights;
	vec R_inv_ys_min_beta;
	vec R_inv_F;
	vec yGEK;
	vec vectorOfF;
	mat correlationMatrixDot;
	mat upperDiagonalMatrixDot;

	double likelihood;
	double beta0;
	double sigmaSquared;

	double epsilonGEK;

	double genErrorGEK;
	int maxNumberOfTrainingIterations;

	void updateWithNewData(void);
	void updateModelParams(void);

	Correlationfunction correlationfunction;

	double computeCorrelation(rowvec x_i, rowvec x_j, vec theta) const;
	double computedR_dxj(rowvec x_i, rowvec x_j,int k) const;
	double computedR_dxi_dxj(rowvec x_i, rowvec x_j, int l,int k) const;
	double computedR_dxi(rowvec x_i, rowvec x_j,int k) const;
	double likelihood_function(vec theta); // Modified by Kai

	void computeCorrelationMatrixDot(vec theta);
	vec computeCorrelationVectorDot(rowvec x) const;

	/* Added by Kai */

    int num;
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


	GEKModel();
	GEKModel(std::string name);

	void setNameOfInputFile(std::string);
	void setNameOfHyperParametersFile(std::string);
	void setNumberOfTrainingIterations(unsigned int);


	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);
	void train(void);
	double interpolateWithGradients(rowvec x) const ;
	double interpolate(rowvec x) const ;

	mat interpolate_all(mat x);

	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;
	void calculateExpectedImprovement(CDesignExpectedImprovement &designCalculated) const;
	void addNewSampleToData(rowvec newsample);


	double calculateExpectedImprovement(rowvec xp);
	double getyMin(void) const;
	vec getKrigingWeights(void) const;
	void setKrigingWeights(vec);
	vec getRegressionWeights(void) const;
	void setRegressionWeights(vec weights);
	void setEpsilon(double inp);
	void setLinearRegressionOn(void);
	void setLinearRegressionOff(void);

	void resetDataObjects(void);
	void resizeDataObjects(void);
	void updateModelWithNewData(mat newData);
	void updateModelWithNewData(void);
	void updateAuxilliaryFields(void);

	/* Hooke Jeeves algorithm */

	void boxmin(vec hyper_lb, vec hyper_ub, int num);
    void start(vec int_hyper, vec hyper_lb, vec hyper_ub, int num);
	void explore(vec int_hyper, double likelihood, int num);
    void move(vec hyper_1, vec hyper_2, double likelihood, int num);

    mat getTheta(void) const;
    vec getLikelihood(void) const;
    vec getOptimalTheta(void) const;
    double getOptimalLikelihood(void) const;

	/* test functions */

	friend void testGEKcalculateRDot(void);
	friend void testGEKcalculateRDotValidateWithWingweight(void);
	friend void testGEKcalculateCorrelationVectorDotWithWingweight(void);
	friend void testGEKWithWingweight(void);
	friend void testGEKValueOfMuWithWingweight(void);
	friend void testGEKPredictionWithWingweight(void);
	friend void testGEKPredictionWithWaves(void);

};

#endif
