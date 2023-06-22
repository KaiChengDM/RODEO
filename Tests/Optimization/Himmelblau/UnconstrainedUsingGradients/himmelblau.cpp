#include<stdio.h>
#include<math.h>
#include <unistd.h>
#include <fstream>

using namespace std;

double HimmelblauAdj(double *x, double *xb) {
	double tempb;
	double tempb0;
	tempb = 2.0*pow(x[0]*x[0]+x[1]-11.0, 2.0-1);
	tempb0 = 2.0*pow(x[0]+x[1]*x[1]-7.0, 2.0-1);
	xb[0] = tempb0 + 2*x[0]*tempb;
	xb[1] = 2*x[1]*tempb0 + tempb;

	return pow( (x[0]*x[0]+x[1]-11.0), 2.0 ) + pow( (x[0]+x[1]*x[1]-7.0), 2.0 );

}

int main(void){

double x[2];
double xb[2];

FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = HimmelblauAdj(x, xb);

/* FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"himmelblau_function = %15.10f\n",result);
fprintf(outp,"himmelblau_gradient = %15.10f, %15.10f\n",xb[0],xb[1]);
fclose(outp);*/

std::ofstream obj_value;
obj_value.open("objFunVal.dat");
obj_value << result << std::endl;
obj_value.close();

std::ofstream obj_grad;
obj_grad.open("objFunGrad.dat");
for (int i=0; i<2;i++){
	   obj_grad << xb[i] << std::endl;
  }
obj_grad.close();


return 0;
}
