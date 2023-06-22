#include<stdio.h>
#include<math.h>
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


double Constraint1(double *x) {
	
	return x[0]*x[0] + x[1]*x[1];

}

double Constraint2(double *x) {
	
	return x[0] + x[1];

}



int main(void){

double x[2];
double xb[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = HimmelblauAdj(x,xb);
double constraint1 = Constraint1(x);
double constraint2 = Constraint2(x);

/*FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"objective_function = %15.10f\n",result);
fprintf(outp,"objective_function_gradient = %15.10f, %15.10f\n",xb[0],xb[1]);
fprintf(outp,"Constraint1 = %15.10f\n", constraint1);
fprintf(outp,"Constraint2 = %15.10f\n", constraint2);
fclose(outp);*/

std::ofstream obj_value;
obj_value.open("objFunVal.dat");
obj_value.precision(15);
obj_value.setf(ios_base::showpoint);
obj_value << result << std::endl;
obj_value.close();

std::ofstream obj_grad;
obj_grad.open("objFunGrad.dat");
obj_grad.precision(15);
obj_grad.setf(ios_base::showpoint);
for (int i=0; i<2;i++){
	   obj_grad << xb[i] << std::endl;
  }
obj_grad.close();

std::ofstream con_value1;
con_value1.open("conFunVal1.dat");
con_value1.precision(15);
con_value1.setf(ios_base::showpoint);
con_value1 << constraint1 << std::endl;
con_value1.close();

std::ofstream con_value2;
con_value2.open("conFunVal2.dat");
con_value2.precision(15);
con_value2.setf(ios_base::showpoint);
con_value2 << constraint2 << std::endl;
con_value2.close();

return 0;
}
