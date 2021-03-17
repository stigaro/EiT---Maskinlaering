import os
import subprocess

# The working directory of this project
__WORKING_DIRECTORY = 'prosjekt'

# Definition of all the repositories we will download
repositories = dict({
    "taco-master": "https://github.com/pedropro/TACO.git"
})

# Checks for correct working directory before performing script
if not os.path.basename(os.getcwd()) == __WORKING_DIRECTORY:
    raise RuntimeError('Script must be run with working directory set at project folder root')

# Clones all of the repositories into their folder names, and deletes the '.git' to make sure they are not repositories anymore.
for repository in repositories:
    subprocess.check_output('git clone ' + repositories[repository] + ' ' + repository, cwd='libraries', shell=True).decode()
    subprocess.run('RD /S /Q .git', cwd='libraries/' + repository, shell=True)
