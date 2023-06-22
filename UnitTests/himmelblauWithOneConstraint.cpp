#include<stdio.h>
#include<math.h>
#include <fstream>

using namespace std;

double Himmelblau(double *x) {

	return pow( (x[0]*x[0]+x[1]-11.0), 2.0 ) + pow( (x[0]+x[1]*x[1]-7.0), 2.0 );
}
double constraint(double *x) {

	return x[0]+ x[1];
}

int main(void){

double x[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = Himmelblau(x);
double constraintVal = constraint(x);

/*FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"himmelblau_function = %15.10f\n",result);
fprintf(outp,"constraint_function = %15.10f\n",constraintVal);
fclose(outp);*/

std::ofstream obj_value;
obj_value.open("objFunVal.dat");
obj_value.precision(15);
obj_value.setf(ios_base::showpoint);
obj_value << result << std::endl;
obj_value.close();


std::ofstream con_value1;
con_value1.open("conFunVal.dat");
con_value1.precision(15);
con_value1.setf(ios_base::showpoint);
con_value1 << constraintVal << std::endl;
con_value1.close();

return 0;
}










