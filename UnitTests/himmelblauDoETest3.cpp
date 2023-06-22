#include<stdio.h>
#include<math.h>
#include <fstream>

using namespace std;

double Himmelblau(double *x) {
	
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
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = Himmelblau(x);
double constraint1 = Constraint1(x);
double constraint2 = Constraint2(x);

/* FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"objective_function = %15.10f\n",result);
fprintf(outp,"Constraint1 = %15.10f\n", constraint1);
fprintf(outp,"Constraint2 = %15.10f\n", constraint2);
fclose(outp); */

  std::ofstream obj_value;
  obj_value.open("objFunVal.dat");
  obj_value.precision(15);
  obj_value.setf(ios_base::showpoint);
  obj_value << result << std::endl;
  obj_value.close();


  std::ofstream const_value1;
  const_value1.open("conFunVal1.dat");
  const_value1.precision(15);
  const_value1.setf(ios_base::showpoint);
  const_value1 << constraint1 << std::endl;
  const_value1.close();

  std::ofstream const_value2;
  const_value2.open("conFunVal2.dat");
  const_value2.precision(15);
  const_value2.setf(ios_base::showpoint);
  const_value2 << constraint2 << std::endl;
  const_value2.close();

return 0;
}
