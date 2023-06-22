#include<stdio.h>
#include<math.h>
#include <stack>
#include <unistd.h>
#include <fstream>

using namespace std;


double EggholderAdj(double *x, double *xb) {    // gradient
	double fabs0;
	double fabs0b;
	double fabs1;
	double fabs1b;
	double temp;
	double temp0;
	int branch;

	xb[0] = 0.0;
	xb[1] = 0.0;

	std::stack<int> intStack;


	if (x[1] + 0.5*x[0] + 47.0 >= 0.0) {
		fabs0 = x[1] + 0.5*x[0] + 47.0;
		intStack.push(1);
	} else {
		fabs0 = -(x[1]+0.5*x[0]+47.0);
		intStack.push(0);
	}
	if (x[0] - (x[1] + 47.0) >= 0.0) {
		fabs1 = x[0] - (x[1] + 47.0);
		intStack.push(0);
	} else {
		fabs1 = -(x[0]-(x[1]+47.0));
		intStack.push(1);
	}
	temp = sqrt(fabs0);
	temp0 = sqrt(fabs1);
	xb[1] = xb[1] - sin(temp);
	fabs0b = (fabs0 == 0.0 ? 0.0 : -(cos(temp)*(x[1]+47.0)/(2.0*
			temp)));
	xb[0] = xb[0] - sin(temp0);
	fabs1b = (fabs1 == 0.0 ? 0.0 : -(cos(temp0)*x[0]/(2.0*temp0)));

	branch = intStack.top();
	intStack.pop();

	if (branch == 0) {
		xb[0] = xb[0] + fabs1b;
		xb[1] = xb[1] - fabs1b;
	} else {
		xb[1] = xb[1] + fabs1b;
		xb[0] = xb[0] - fabs1b;
	}
	branch = intStack.top();
	intStack.pop();

	if (branch == 0) {
		xb[0] = xb[0] - 0.5*fabs0b;
		xb[1] = xb[1] - fabs0b;
	} else {
		xb[1] = xb[1] + fabs0b;
		xb[0] = xb[0] + 0.5*fabs0b;
	}

	return -(x[1]+47.0)*sin(sqrt(fabs(x[1]+0.5*x[0]+47.0)))-x[0]*sin(sqrt(fabs(x[0]-(x[1]+47.0) )));
}


int main(void){

double x[2];
double xb[2];

FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = EggholderAdj(x, xb);

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
