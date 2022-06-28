#include<stdio.h>
#include<math.h>

double DropwaveAdj(double *x) {

	return -(1+cos(12*sqrt(x[0]*x[0]+x[1]*x[1])))/(0.5*(x[0]*x[0]+x[1]*x[1])+2);

}

int main(void){

double x[2];

FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = DropwaveAdj(x);
FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"Dropwave_function = %15.10f\n",result);
fclose(outp);

return 0;
}
