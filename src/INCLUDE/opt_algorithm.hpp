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
 * Authors: Kai Cheng from SDU
 *
 */

#ifndef OPT_ALGORITHM_HPP
#define OPT_ALGORITHM_HPP

#include <armadillo>
#include "objective_function.hpp"
#include "constraint_functions.hpp"
#include "random_functions.hpp"
#include "optimization.hpp"
// #include "Galgo.hpp"       // Genetic algorithm taken from https://github.com/olmallet81/GALGO-2.0

using namespace arma;

class Hooke_Jeeves {        // Hooke-Jeeves pattern search method for optimization

   int num =1 ;

   vec dv_lb;
   vec dv_ub;
   vec dv_in;

   mat dv_cur;
   vec dv_par;
   vec dv_optimal;

   vec  increment;
   uvec ind_increment;

   mat hyperoptimizationHistory;

   unsigned int numberOfIteration;

   int dim;

   vec obj_cur;
   double obj_optimal;
   double obj;

public:

    Hooke_Jeeves();

    void boxmin(mat dv, vec dv_lb, vec dv_ub);
    void start(vec dv_in, vec dv_lb, vec dv_ub, int num);
    void explore(vec dv_in, double obj, int num);
    void move(vec dv_1, vec dv_2, double obj, int num);

    mat get_dv(void) ;
    vec get_obj(void) ;
    vec getOptimal_dv(void) ;
    double getOptimal_obj(void);

    double objectiveFunction(vec dv);   // objective function

};


class Genetic_Algorithm {      //  Genetic Algorithm for optimization

   int dim;

   double obj_optimal;

   const vec lowerBound;
   const vec upperBound;

   const int populationSize = 50;
   const int maxIterations = 100;

public:


    Genetic_Algorithm();

    double objectiveFunction(const vec & x);

    void constraintFunction(vec & x, vec lowerBound, vec upperBound);


};



#endif
