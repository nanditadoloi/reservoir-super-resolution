import os
import glob

exclude_files = []

dataFilenamesList = glob.glob('*.DATA')

for datafile in dataFilenamesList:
    if not (datafile in exclude_files):
        os.system("mpirun --allow-run-as-root -np 10 flow " + datafile)
