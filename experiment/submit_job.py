import os
from time import sleep

def habanero_script(name, command, hours=5, memory=1):

    job_str = """#!/bin/sh
#
#SBATCH --account=dsi            # The account name for the job.
#SBATCH --job-name=%s         # The job name.
#SBATCH --error=log/%s.e
#SBATCH --output=log/%s.o
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=%s:00:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=%sgb        # The memory the job will use per cpu core.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liping.liulp@gmail.com

%s

# End of script""" % (name, name, name, hours, memory, command)
    return job_str


for K in [16, 32, 64]:
    for fold in xrange(10):
    
        command = '''
python -c "import experiment; experiment.fold_learn(cont_train=False, K=%d, sigma2ar=100, sigma2b=100, link_func='softplus', intercept_term=False, scale_context=False, normalize_context=False, downzero=False, use_obscov=False, zeroweight=1.0, fold=%d)"''' % (K, fold)
    
        name = 'bd_k%df%d' % (K, fold)
        script = habanero_script(name, command, hours=10, memory=1)
    
        print('-----------------------------------------------')
        print(script)
        with open('job.sh', 'w') as jfile:
            jfile.write(script)
    
        os.system('sbatch job.sh')
        sleep(0.3)


