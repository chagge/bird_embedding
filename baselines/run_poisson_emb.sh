#!/bin/bash

echo "Submit the pemb job to the system: "
../lib/poisson_efe/pemb -dir ../data/subset_pa_201407/data_folds/0/ -K 10 -max-iterations 2000 -Nthreads 3 -zeroFactor 0.1 -eta 0.01 -s2alpha 1000 -s2rho 1000 > job0.log 

