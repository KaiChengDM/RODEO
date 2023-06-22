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




int main(void){

double x[2];
double xb[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = Himmelblau(x);
double constraint1 = Constraint1(x);

/*FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"objective_function = %15.10f\n",result);
fprintf(outp,"Constraint1 = %15.10f\n", constraint1);
fclose(outp);*/

std::ofstream obj_value;
obj_value.open("objFunVal.dat");
obj_value.precision(15);
obj_value.setf(ios_base::showpoint);
obj_value << result << std::endl;
obj_value.close();
std::ofstream con_value1;

con_value1.open("conFunVal1.dat");
con_value1.precision(15);
con_value1.setf(ios_base::showpoint);
con_value1 << constraint1 << std::endl;
con_value1.close();

return 0;
}
