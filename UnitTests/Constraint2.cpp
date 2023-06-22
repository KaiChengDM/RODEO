#include<stdio.h>
#include<math.h>
#include <fstream>

using namespace std;

double constraint2(double *x) {

	return x[0] * x[1];

}



int main(void){

double x[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double constraintVal = constraint2(x);
/*FILE *outp = fopen("constraint2.dat","w");
fprintf(outp,"constraint_function2 = %15.10f\n",constraintVal);
fclose(outp);*/

std::ofstream con_value2;
con_value2.open("conFunVal2.dat");
con_value2.precision(15);
con_value2.setf(ios_base::showpoint);
con_value2 << constraintVal << std::endl;
con_value2.close();

return 0;
}
