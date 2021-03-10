import os
import subprocess

__WORKING_DIRECTORY = 'prosjekt'
__TACO_GIT_REPOSITORY = "https://github.com/pedropro/TACO.git"

# Checks for correct working directory before performing script
if not os.path.basename(os.getcwd()) == __WORKING_DIRECTORY:
    raise RuntimeError('Script must be run with working directory set at project folder root')

subprocess.check_output('git clone ' + __TACO_GIT_REPOSITORY + ' taco-master', cwd='libraries', shell=True).decode()
