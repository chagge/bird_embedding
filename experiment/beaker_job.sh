#!/bin/bash

for i in `seq 0 4`;
do
    echo "Submit job for fold $i"
    nohup python -c "import experiment; experiment.fold_learn(K=10, sigma2ar=100, sigma2b=100, link_func='softplus', intercept_term=True, scale_context=False, normalize_context=False, downzero=False, use_obscov=False, zeroweight=1.0, fold=$i)" > log/jobexp$i.log &
done




