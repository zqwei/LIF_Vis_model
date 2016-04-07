from glob import glob
from os.path import exists, splitext
# from os import system

num_core = 30
ncol = 2

with open ('run_batch.bat', 'w') as run_file:
    run_file.write('set PATH=c:\\nrn\\bin;%PATH%\n\n')
    for n_run in glob("output_*"):
        if not exists(n_run+'/tot_f_rate.dat'):
            sfile = splitext(n_run)[0][7:]
            run_file.write('mpiexec -np %d nrniv -mpi run_%s.py > output_%s/log.txt\n' % (num_core, sfile, sfile))
