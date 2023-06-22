#include<stdio.h>
#include<math.h>
#include <fstream>

using namespace std;


double Rosenbrock8D(double *x){

	double temp=0.0;

	for(int i=0; i<9; i++){

		temp+= 100.0*(x[i+1]-x[i]*x[i])*(x[i+1]-x[i]*x[i]) + (1-x[i])*(1-x[i]);
	}
	return temp;

}


int main(void){

double x[10];

FILE *inp = fopen("dv.dat","r");

for (unsigned int k = 0; k < sizeof(x); k++ ){
	fscanf(inp,"%lf",&x[k]);
}
fclose(inp);


double result = Rosenbrock8D(x);

std::ofstream obj_value;
obj_value.open("objFunVal.dat");
obj_value << result << std::endl;
obj_value.close();


return 0;
}


