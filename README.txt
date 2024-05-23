README
—————-

Cython files first need to be installed with the setup.py script. 
Requirements are included in the requirements.txt file. 'geomappy' needs to be installed separately.

run_models.py runs the whole modelling framework

The data required are not shared here, but can be found in online repositories, reported in the 
paper. They need to be structured according to the interface provided in src/data.py, or otherwise
the pointers in src/data.py need to be adapted. All reprojections are taken care of internally.
