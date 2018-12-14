import os
import shutil

from pygit2 import Repository
runName = Repository('.').head.shorthand
modelSaveDir = os.path.join(os.getcwd(), 'output', runName)
shutil.rmtree(modelSaveDir)