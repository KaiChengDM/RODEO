#include<stdio.h>
#include<math.h>
#include <unistd.h>
#include <iostream>
#include <armadillo>
using namespace arma;
using namespace std;

double Rosenbrock10DAdj(double *x, double *xb) {

	double temp = 0.0;
	double tempb0 = 0.0;


	for(int i=0; i<10; i++){
		xb[i]=0.0;

	}

	for(int i=0; i<9; i++){

		temp+= 100.0*(x[i+1]-x[i]*x[i])*(x[i+1]-x[i]*x[i]) + (1-x[i])*(1-x[i]);
	}

	tempb0 = 1.0;
	{
		double tempb;

		for (int i = 0; i < 9; i++) {
			tempb = 100.0*2*(x[i+1]-x[i]*x[i]);
			xb[i] = - 2*(1-x[i]) - 2*x[i]*tempb;
		}

		xb[9] = 100.0*2*(x[9]-x[8]*x[8]);

	}

	return temp;
}

double constraint(double *x, double *xb1){

	    double sum = 0;
		for(int i=0; i<10; i++){
				xb1[i]=1;
                sum = sum + x[i];
	    }

		return sum;
}


int main(void){

  double x[10];
  double xb[10];
  double xb1[10];

  FILE *inp = fopen("dv.dat","r");

  for (unsigned int k = 0; k < sizeof(x); k++ ){
		fscanf(inp,"%lf",&x[k]);
   }
  fclose(inp);

  double result = Rosenbrock10DAdj(x, xb);
  double constraintValue = constraint(x,xb1);

  /*FILE *outp = fopen("objFunVal.dat","w");
  fprintf(outp,"Rosenbrock_function = %15.10f\n",result);
  fprintf(outp,"Rosenbrock_gradient = %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f\n",xb[0],xb[1],xb[2],xb[3],xb[4],xb[5],xb[6],xb[7],xb[8],xb[9]);
  fclose(outp);

  FILE *outp1 = fopen("conFunVal.dat","w");
  fprintf(outp1,"constraint_function = %15.10f\n",constraintValue);
  fprintf(outp1,"constraint_gradient = %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f\n",xb1[0],xb1[1],xb1[2],xb1[3],xb1[4],xb1[5],xb1[6],xb1[7],xb1[8],xb1[9]);
  fclose(outp1);*/

   std::ofstream obj_value;
   obj_value.open("objFunVal.dat");
   obj_value << result << std::endl;
   obj_value.close();

   std::ofstream obj_grad;
   obj_grad.open("objFunGrad.dat");
   for (int i=0; i<10;i++){
	   obj_grad << xb[i] << std::endl;
     }
   obj_grad.close();

   std::ofstream const_value;
   const_value.open("conFunVal.dat");
   const_value << constraintValue << std::endl;
   const_value.close();

   std::ofstream const_grad;
   const_grad.open("conFunGrad.dat");
   for (int i=0; i<10;i++){
 	  const_grad << xb1[i] << std::endl;
   }
   const_grad.close();

   std::ofstream const_value1;
   const_value1.open("conFunVal1.dat");
   const_value1 << constraintValue+1 << std::endl;
   const_value1.close();

   std::ofstream const_grad1;
   const_grad1.open("conFunGrad1.dat");
   for (int i=0; i<10;i++){
   	  const_grad1 << xb1[i] << std::endl;
   }
   const_grad1.close();

  return 0;

}

