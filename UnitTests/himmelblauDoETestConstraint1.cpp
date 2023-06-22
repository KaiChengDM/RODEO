#include<stdio.h>
#include<math.h>
#include <fstream>

using namespace std;

double Constraint2(double *x) {
	
	return x[0]+ x[1];

}




int main(void){

double x[2];
double xb[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double constraint2 = Constraint2(x);

/*FILE *outp = fopen("constraint2.dat","w");
fprintf(outp,"Constraint2 = %15.10f\n", constraint2);
fclose(outp);*/

std::ofstream con_value2;
con_value2.open("conFunVal2.dat");
con_value2.precision(15);
con_value2.setf(ios_base::showpoint);
con_value2 << constraint2 << std::endl;
con_value2.close();

return 0;
}
