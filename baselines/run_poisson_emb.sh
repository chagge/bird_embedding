#!/bin/bash

echo "Submit the pemb job to the system: "
nohup ../lib/poisson_efe/pemb -dir ../data/subset_pa_201407/ -K 10 -max-iterations 10000 -Nthreads 3 > job.log &
