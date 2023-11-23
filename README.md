# CGCompiler: Automated Coarse-Grained Molecule Parametrization via Noise-Resistant Mixed-Variable Optimization

CGCompiler is a Python tool that assists in coarse-grained molecule parametrization and is aimed mostly at Martini 3, but can be used with other force fields as well. CGCompiler uses mixed-variable particle swarm optimization (PSO) to optimize force field parameters. Candidate parametrizations are scored by running MD simulations and measuring how well certain target observables are reproduced. For a more detailed explanation have a look at our paper: https://doi.org/10.1021/acs.jctc.3c00637


## Installation
CGCompiler does not need to be installed, just put the files where you want to work with it. However, CGCompiler does have a couple of requirements. As simulations are run with GROMACS, you need a recent version of it (CGCompiler has been tested with GROMACS versions 2020.x, 2021.x, 2022.x).


## Usage
In its current form this repository provides an example on how to use CGCompiler to parametrize the lipid sphingomyelin, very similar to what was done in the paper. To parametrize a different lipid with the same observables, only a few changes have to be made. The general workflow is like this:

1. provide reference data, which needs to be stored in `user/ref_data/system_name/`
2. provide base simulation files, which need to be stored in `user/base/production/system_name/`
3. provide a base ITP file, with the molecule's mapping and bonded interactions defined
4. adapt `usersettings.py` to your parametrization task:
    + specify the training systems
    + specify the path to the base ITP
    + provide information about the molecule
    + specify what to optimize, and set parameter ranges and feasible bead types
    + specify paths to reference data, or enter target values (like the melting temperature) in `usersettings.py` itself
    + specify which target observables to calculate and set weights
    + set some simulation parameters
5. set swarm size and number of iterations in run.py
6. run with `mpirun -np #ranks python run.py -n #processes_per_rank`, to have everything run in parallel set ranks = swarm_size * number of training systems
7. after every iteration CGCompiler writes a checkpoint from which the parametrization can be restarted, the path to the checkpoint file should be specified in `run.py`
8. analyze parametrization results with the provided jupyter notebook
9. validate the new molecule

If you want to add new observables, you will new to write a Python function, e.g., in `observables.py`, make changes to `compute_observables.py`, and specify weights and what scoring function to use on the observable in `usersettings.py`



## Citation
If you use CGCompiler please cite the following article: Kai Steffen Stroh, Paulo C. T. Souza, Luca Monticelli and Herre Jelger Risselada; CGCompiler: Automated Coarse-Grained Molecule Parametrization via Noise-Resistant Mixed-Variable Optimization, J. Chem. Theory Comput. 2023; doi: https://doi.org/10.1021/acs.jctc.3c00637


    @Article{CGCompiler2023,
    author    = {Stroh, Kai Steffen and Souza, Paulo C. T. and Monticelli, Luca and Risselada, Herre Jelger},
    journal   = {J. Chem. Theory Comput.},
    title     = {CGCompiler: Automated Coarse-Grained Molecule Parametrization via Noise-Resistant Mixed-Variable Optimization},
    year      = {2023},
    issn      = {1549-9626},
    doi       = {10.1021/acs.jctc.3c00637},
    publisher = {American Chemical Society (ACS)},
    }