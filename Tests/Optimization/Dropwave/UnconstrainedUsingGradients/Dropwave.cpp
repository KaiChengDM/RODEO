#include<stdio.h>
#include<math.h>
#include <unistd.h>
#include <fstream>

using namespace std;

double DropwaveAdj(double *x, double *xb) {

	xb[0] = (x[0]*(cos(12*sqrt(x[0]*x[0] + x[1]*x[1])) + 1))/pow((x[0]*x[0]/2 + x[1]*x[1]/2 + 2),2) + (12*x[0]*sin(12*sqrt(x[0]*x[0] + x[1]*x[1])))/(sqrt(x[0]*x[0] + x[1]*x[1])*(x[0]*x[0]/2 + x[1]*x[1]/2 + 2));

	xb[1] = (x[1]*(cos(12*sqrt(x[0]*x[0] + x[1]*x[1])) + 1))/pow((x[0]*x[0]/2 + x[1]*x[1]/2 + 2),2) + (12*x[1]*sin(12*sqrt(x[0]*x[0] + x[1]*x[1])))/(sqrt(x[0]*x[0] + x[1]*x[1])*(x[0]*x[0]/2 + x[1]*x[1]/2 + 2));

	return -(1+cos(12*sqrt(x[0]*x[0]+x[1]*x[1])))/(0.5*(x[0]*x[0]+x[1]*x[1])+2);

}

int main(void){

double x[2];
double xb[2];

FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = DropwaveAdj(x, xb);

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
