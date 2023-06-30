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


#include "kriging_training.hpp"
#include "matrix_vector_operations.hpp"
#include "test_functions.hpp"
#include<gtest/gtest.h>


TEST(testKriging, testKrigingConstructor){

	KrigingModel testModel("testKrigingModel");
	std::string filenameDataInput = testModel.getNameOfInputFile();
	ASSERT_TRUE(filenameDataInput == "testKrigingModel.csv");
	ASSERT_FALSE(testModel.areGradientsOn());


}

TEST(testKriging, testReadDataAndNormalize){

	TestFunction testFunctionEggholder("Eggholder",2);

	testFunctionEggholder.setFunctionPointer(Eggholder);

	testFunctionEggholder.setBoxConstraints(0,200.0);
	mat samples = testFunctionEggholder.generateRandomSamples(10);
	saveMatToCVSFile(samples,"Eggholder.csv");


	KrigingModel testModel("Eggholder");


	testModel.readData();
	testModel.setBoxConstraints(0.0, 200.0);

	testModel.normalizeData();

	unsigned int N = testModel.getNumberOfSamples();
	ASSERT_TRUE(N == 10);
	unsigned int d = testModel.getDimension();
	ASSERT_TRUE(d == 2);
	mat rawData = testModel.getRawData();
	bool ifBothMatricesAreEqual = isEqual(samples,rawData,10E-10);


	ASSERT_TRUE(ifBothMatricesAreEqual);

	remove("Eggholder.csv");
}



TEST(testKriging, testsetParameterBounds){


	TestFunction testFunctionEggholder("Eggholder",2);

	testFunctionEggholder.setFunctionPointer(Eggholder);

	testFunctionEggholder.setBoxConstraints(0,200.0);
	mat samples = testFunctionEggholder.generateRandomSamples(10);
	saveMatToCVSFile(samples,"Eggholder.csv");


	KrigingModel testModel("Eggholder");


	testModel.readData();

	Bounds boxConstraints(2);
	boxConstraints.setBounds(0.0,2.0);

	testModel.setBoxConstraints(boxConstraints);


}




TEST(testKriging, testInSampleErrorCloseToZeroWithoutTraining){


	mat samples(10,3);

	/* we construct first test data using the function x1*x1 + x2 * x2 */
	for (unsigned int i=0; i<samples.n_rows; i++){
		rowvec x(3);
		x(0) = generateRandomDouble(-1.0,2.0);
		x(1) = generateRandomDouble(-1.0,2.0);

		x(2) = x(0)*x(0) + x(1)*x(1);
		samples.row(i) = x;

	}


	vec lb(2); lb.fill(-1.0);
	vec ub(2); ub.fill(2.0);
	saveMatToCVSFile(samples,"KrigingTest.csv");

	KrigingModel testModel("KrigingTest");
	testModel.readData();
	testModel.setBoxConstraints(lb, ub);
	testModel.normalizeData();
	testModel.initializeSurrogateModel();

	double mean_y = testModel.readOutputMean();
	double std_y =  testModel.readOutputStd();

	rowvec xp(2); xp(0) = samples(0,0); xp(1) = samples(0,1);

	rowvec xpnorm = normalizeRowVector(xp,lb,ub);

	double ftilde = testModel.interpolate(xpnorm);

	ftilde = ftilde *std_y+mean_y;

	double error = fabs(ftilde - samples(0,2));
	EXPECT_LT(error, 10E-6);


}

TEST(testKriging, testInSampleErrorCloseToZeroAfterTraining){


	mat samples(10,3);

	/* we construct first test data using the function x1*x1 + x2 * x2 */
	for (unsigned int i=0; i<samples.n_rows; i++){
		rowvec x(3);
		x(0) = generateRandomDouble(-1.0,2.0);
		x(1) = generateRandomDouble(-1.0,2.0);

		x(2) = x(0)*x(0) + x(1)*x(1);
		samples.row(i) = x;

	}

	vec lb(2); lb.fill(-1.0);
	vec ub(2); ub.fill(2.0);
	saveMatToCVSFile(samples,"KrigingTest.csv");

	KrigingModel testModel("KrigingTest");
	testModel.readData();
	testModel.setBoxConstraints(lb, ub);
	testModel.normalizeData();
	testModel.initializeSurrogateModel();
	testModel.train();

	double mean_y = testModel.readOutputMean();
	double std_y =  testModel.readOutputStd();

	testModel.setNumberOfTrainingIterations(100);

	rowvec xp(2); xp(0) = samples(0,0); xp(1) = samples(0,1);

	rowvec xpnorm = normalizeRowVector(xp,lb,ub);

	double ftilde = testModel.interpolate(xpnorm);
	ftilde = ftilde *std_y+mean_y;
	double error = fabs(ftilde - samples(0,2));
	EXPECT_LT(error, 10E-6);


}


TEST(testKriging, testPOD){

   // Here we test reduced order model using proper orthogonal decomposition + Kriging

    int m = 1000;

	mat samples(10,2+m);

	vec t = linspace(0,m-1,m);
	vec xrow(2+m);

	/* we construct first test data using the function x1*x1 + x2 * x2 */
	for (unsigned int i=0; i<samples.n_rows; i++){

		vec x(2);

		x(0) = generateRandomDouble(-1.0,2.0);
		x(1) = generateRandomDouble(-1.0,2.0);

		//x(2) = x(0)*x(0) + x(1)*x(1);

		xrow.subvec(0,1) = x;

		xrow.subvec(2,1+m) = (x(0)*x(0) + x(1)*x(1))*t/m;

		samples.row(i) = xrow.t();

	}

	vec lb(2); lb.fill(-1.0);
	vec ub(2); ub.fill(2.0);

	saveMatToCVSFile(samples,"KrigingTest.csv");

	KrigingModel testModel("KrigingTest");

	testModel.setVectorConstraintOn();
	testModel.setConstraintLength(m);

	testModel.readData();
	testModel.setBoxConstraints(lb, ub);
	testModel.normalizeData();
	testModel.initializeSurrogateModel();


    int rank = testModel.readRank();

     mat basis = testModel.getPodBasis();

     vec mean_vec1(m);
	 vec variance_vec1(m);

	 rowvec meanvector = testModel.readOutputMeanVector();
	 rowvec stdvector  = testModel.readOutputStdVector();

	 rowvec xp(2); xp(0) = samples(0,0); xp(1) = samples(0,1);

	 rowvec xpnorm = normalizeRowVector(xp,lb,ub);

	 // vec fTilde = testModel.interpolate_vec(xp);

	 vec ftildeOutput; vec sSqrOutput;

	 testModel.interpolateWithVariance_vec(xpnorm,ftildeOutput,sSqrOutput);

	  mean_vec1.zeros();   variance_vec1.zeros();

	  for (unsigned int i = 0; i< rank; i++ ){       // recover the full state solution

	  	  if (stdvector(i) != 0){

	  				  ftildeOutput(i) = ftildeOutput(i)*stdvector(i) + meanvector(i);

	  				  sSqrOutput(i) = sSqrOutput(i)*stdvector(i)*stdvector(i);

	  	   }else {

	  				  ftildeOutput(i) = ftildeOutput(i) + meanvector(i);

	  				  sSqrOutput(i) = sSqrOutput(i)*stdvector(i)*stdvector(i);

	  			  }

	  	   mean_vec1 = mean_vec1 + basis.col(i)*(ftildeOutput(i));

	  	   variance_vec1 = variance_vec1 + (basis.col(i) % basis.col(i))*(sSqrOutput(i));

	  	}


	 rowvec firstvec = samples.submat(0,2,0,m+1);

	 double error = sum((mean_vec1 - firstvec.t())%(mean_vec1 - firstvec.t()));

	 double variance = sum(variance_vec1);

	 EXPECT_LT(error, 10E-8);
	 EXPECT_LT(variance, 10E-8);


}

