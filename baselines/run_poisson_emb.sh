#!/bin/bash

echo "Submit the pemb job to the system: "

data_dir=../data/subset_pa_201407
fold=1

../lib/poisson_efe/pemb -dir $data_dir/data_folds/$fold/ -outdir $data_dir/result/pemb -K 10 -max-iterations 100 -Nthreads 4 -eta 0.01 -zeroFactor 0 -s2alpha 1 -s2rho 1 -avgContext 1 -rfreq 10000 > job0.log 
#../lib/poisson_efe/pemb -dir ../data/subset_pa_201407/data_folds/0/ -K 10 -max-iterations 2000 -Nthreads 3 -zeroFactor 0.1 -eta 0.01 -s2alpha 1 -s2rho 1 > job0.log 

