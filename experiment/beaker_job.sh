#!/bin/bash

for i in `seq 0 9`;
do
    echo "Submit job for fold $i"
    nohup python -c "import experiment; experiment.fold_learn(K=10, sigma2ar=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True, fold=$i)" > log/job$i.log &
done




