import os
import shutil
import subprocess

from sources.utility import Constant

# Definition of all the repositories we will download
repositories = dict({
    "taco_master": "https://github.com/pedropro/TACO.git",
    "vision_master": "https://github.com/pytorch/vision.git"
})

# Checks for correct working directory before performing script
if not os.path.basename(os.getcwd()) == Constant.WORKING_DIRECTORY:
    raise RuntimeError('Script must be run with working directory set at project folder root')

# Clones all of the repositories into their folder names, and deletes the '.git' to make sure they are not repositories anymore.
# We will also delete the folder if they already exist.from.
for repository in repositories:
    shutil.rmtree('libraries' + '/' + repository, ignore_errors=True)
    subprocess.check_output('git clone ' + repositories[repository] + ' ' + repository, cwd='libraries', shell=True).decode()
    subprocess.run('RD /S /Q .git', cwd='libraries/' + repository, shell=True)
