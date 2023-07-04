#include<stdio.h>
#include<math.h>
#include <fstream>
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


int main(void){

double x[10];
double xb[10];

FILE *inp = fopen("dv.dat","r");

for (unsigned int k = 0; k < sizeof(x); k++ ){
	fscanf(inp,"%lf",&x[k]);
}
fclose(inp);

double result = Rosenbrock10DAdj(x, xb);


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

/*FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"Rosenbrock_function = %15.10f\n",result);
fprintf(outp,"Rosenbrock_gradient = %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f, %15.10f\n",xb[0],xb[1],xb[2],xb[3],xb[4],xb[5],xb[6],xb[7],xb[8],xb[9]);
fclose(outp);*/

return 0;
}



