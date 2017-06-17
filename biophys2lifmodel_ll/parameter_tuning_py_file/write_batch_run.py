from glob import glob
import platform
from os.path import splitext

num_core = 30

if len(glob('run_*.py')) > 0:
    if platform.system() == "Windows":
        run_batch_file = 'run_all_py_file.bat'
    else:
        run_batch_file = 'run_all_py_file.sh'
    with open(run_batch_file, 'w') as run_file:
        if platform.system() == "Windows":
            run_file.write('set PATH=c:\\nrn\\bin;%PATH%\n\n')
        for file_name in glob('run_*.py'):
            sfile = splitext(file_name)[0][4:]
            run_file.write('mpiexec -np %d nrniv -mpi run_%s.py > output_%s/log.txt\n'%(num_core, sfile, sfile))