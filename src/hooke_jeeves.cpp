/*
 * RoDeO, a Robust Design Optimization Package
 *
 * This file is part of RoDeO
 *
 * RoDeO is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * RoDeO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Kai Cheng (SDU)
 *
 */

#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>
#include "auxiliary_functions.hpp"
#include "kriging_training.hpp"
#include "aggregation_model.hpp"
#include "hooke_jeeves.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "test_functions.hpp"
#include "optimization.hpp"
#include "lhs.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;


Hooke_Jeeves::Hooke_Jeeves(){};


double Hooke_Jeeves::evaluate(vec dv){

	return 0;                                       // Need to insert the objective function

}

void Hooke_Jeeves::boxmin(mat dv, vec dv_l, vec dv_u){

	dim = size(dv_lb,0);

	vec obj_value = ones(num);

    dv_cur  = dv;
    obj_cur = obj_value;

	/*
	  vec log_ub = log10(dv_u);
	  vec log_lb = log10(dv_l);
      vec random;  random.randu(num);

	  for (unsigned int i=0;i<dim;i++){
	  for (unsigned int j=0;j<num;j++){
		  dv(i,j) = pow(10,random(j)*(log_ub(i)-log_lb(i))+log_lb(i));  // multiple random start points
	  }
    }*/


	dv_lb  = dv_l;    //  lower bound
	dv_ub  = dv_u;    //  upper bound

	//#pragma omp parallel for

	for (unsigned int kk=0;kk<num;kk++){      // Multi-starts

	  start(dv.col(kk),dv_lb,dv_ub,kk);

	  int kmax;

	  if (dim < 2)
	      { kmax = 2;}
      else
	      { kmax = std::min(dim,4);}

	  for (unsigned int k = 0; k < kmax; k++){  // Iterate for kmax times

	   vec dv1 = dv_cur.col(kk);

	   explore(dv_cur.col(kk),obj_cur(kk),kk);

	   move(dv1,dv_cur.col(kk),obj_cur(kk),kk);

	   // cout << "current likelihood is " << likelihood_cur(kk) << endl;

	 }

	}

	obj_value = get_obj();
	dv = get_dv();

	uword i = obj_value.index_min();

	obj_optimal = obj_value(i);
	dv_optimal  = dv.col(i);

}

void Hooke_Jeeves::start(vec dv_in1, vec dv_lb, vec dv_ub, int kk){

	  vec m = linspace(1,dim,dim)/(dim+2);
	  increment = zeros(dim);

	  for (unsigned int k = 0; k < dim; k++){
		  increment(k) = pow(2,m(k));
	   }

	  ind_increment = find(increment != 1);

	  dv_cur.col(kk) = dv_in1;

	  obj_cur(kk) = evaluate(dv_cur.col(kk));

	  numberOfIteration = 0;
      hyperoptimizationHistory = zeros(dim+2,200*dim);

	  hyperoptimizationHistory.col(numberOfIteration) = join_cols(dv_cur.col(kk), vec {obj_cur(kk), 1.0} );


}

void Hooke_Jeeves::explore(vec dv_1, double obj_1, int kk){

	unsigned int j; double DD;  unsigned int atbd;

    dv_cur.col(kk) = dv_1; obj_cur(kk) = obj_1;

	for (unsigned int k = 0; k < size(ind_increment,0); k++){

	   j = ind_increment(k);
	   dv_par = dv_cur.col(kk);
       DD = increment(j);

       if (dv_cur(j,kk) == dv_ub(j)){

    	   atbd = 1;
    	   dv_par(j) =  dv_cur(j,kk)/sqrt(DD); }

       else if (dv_cur(j,kk) == dv_lb(j)){

    	   atbd = 1;
    	   dv_par(j) =  dv_cur(j,kk)*sqrt(DD); }

       else  {

    	   atbd = 0;
    	   dv_par(j) = std::min(dv_ub(j),dv_cur(j,kk)*DD);
       }

       obj = evaluate(dv_par);
       numberOfIteration++;
       hyperoptimizationHistory.col(numberOfIteration)= join_cols(dv_par, vec {obj, 2});

       if (obj < obj_cur(kk)){
            obj_cur(kk) = obj;
            dv_cur.col(kk) = dv_par;  }
       else  {

    	    hyperoptimizationHistory(dim+1,numberOfIteration)= -2;

    	   if (!atbd) {
    		    dv_par(j) = std::max(dv_lb(j),dv_cur(j,kk)/DD);
    	        obj = evaluate(dv_par);

    	        numberOfIteration++;
    	        hyperoptimizationHistory.col(numberOfIteration) = join_cols(dv_par, vec {obj, 2});

    	        if (obj < obj_cur(kk)){
    	        	obj_cur(kk) = obj;
    	        	dv_cur.col(kk) = dv_par;  }
    	        else
    	            hyperoptimizationHistory(dim+1,numberOfIteration)= -2;
    	    }
	     }
	  }

}

void Hooke_Jeeves::move(vec dv_old,vec dv_new, double obj_new, int kk){

	   vec v  = dv_new/dv_old;
	   vec v1 = v-ones(dim,1);

       if (v1.is_zero()){

    	   vec ind = linspace(1,dim,dim);
    	   ind(dim-1) = 0;

    	   for (unsigned int k = 0; k < dim; k++){
    	  	    increment(k) = pow(increment(ind(k)),0.2);
    	    }

    	   obj_cur(kk) = obj_new;
    	   dv_cur.col(kk) = dv_new;

            return ;
        }

        unsigned int rept = 1;   obj_cur(kk) = obj_new;  dv_cur.col(kk) = dv_new;

        while (rept){

		   dv_par = min(join_rows(dv_ub,max(join_rows(dv_lb,dv_new % v),1)),1);
		   obj = evaluate(dv_par);
		   numberOfIteration++;
		   hyperoptimizationHistory.col(numberOfIteration)= join_cols(dv_par, vec { obj, 3});

		   if (obj < obj_cur(kk)){
			   dv_cur.col(kk) = dv_par;
			   obj_cur(kk) = obj;
			   v = v % v;
		   }

		   else {
			   hyperoptimizationHistory(dim+1,numberOfIteration)= -3;
			   rept = 0;

		   }

		   if (size(find(dv_par - dv_ub),0)+size(find(dv_par - dv_ub),0) < 2*dim)
			    rept  =  0;
        }

         vec ind = linspace(1,dim,dim);
         ind(dim-1) = 0;

         for (unsigned int k = 0; k < dim; k++){
          	 increment(k) = pow(increment(ind(k)),0.25);
          }
}

mat Hooke_Jeeves::get_dv(void) {
	return dv_cur;
}

vec Hooke_Jeeves::getOptimal_dv(void) {
	return dv_optimal;
}


vec Hooke_Jeeves::get_obj(void) {
	return obj_cur;
}

double Hooke_Jeeves::getOptimal_obj(void){
	return obj_optimal;
}




