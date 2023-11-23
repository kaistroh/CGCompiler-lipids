"""
Executable wrapper script. 
Initializes particle swarm optimization or genetic algorithm and defines the fitness function.
"""

import sys
import os
#import subprocess
import copy
#import numpy as np


import core.settings            as CoreSettings
import core.simulation          as Sim
import core.system              as System
import core.pso.particle_swarm  as PSO

import user.usersettings        as UserSettings
import user.analysis.scoring_functions as ScoringFunctions

from core.system                import mprint, line_prepender


def _generate_itp(particle, molkey, dir):
    itp = copy.deepcopy(UserSettings.baseITPs[molkey])
    X_disc = particle.X_disc   # dictionary {"beadname": beadtype}
    X_cont = particle.X_cont   # dictionary {"bondname": [b0, fc], "anglename": [a0, fc]}
    for beadname in X_disc:
        itp.changeAtomType(molkey, beadname, X_disc[beadname])

    for bondname in UserSettings.bonds_to_optimize[molkey]:
        itp.changeBond(molkey, bondname, X_cont[bondname][0], X_cont[bondname][1])

    for anglename in UserSettings.angles_to_optimize[molkey]:
        itp.changeAngle(molkey, anglename, X_cont[anglename][0], X_cont[anglename][1])

    itp.write(os.path.join(dir, '%s.itp' %molkey))
    
    line_prepender(os.path.join(dir, '%s.itp' %molkey), '; %s' %particle.id)
    line_prepender(os.path.join(dir, '%s.itp' %molkey), '; %s' %particle.uid)



def _setup_simulation_directory(particle, molkey, training_system, path_parent_dir, dirname=None):
    if dirname is None:
        dirname = particle.uid
    uid_dir = os.path.join(path_parent_dir, dirname)
    System.mkdir(uid_dir)
    System.rm_dir_content(uid_dir) # Ensure folder is empty 

    molkey_dir = os.path.join(uid_dir, molkey)
    System.mkdir(molkey_dir)
    tr_system_dir = os.path.join(molkey_dir, training_system)
    System.mkdir(tr_system_dir)
    _generate_itp(particle, molkey, tr_system_dir)

    return tr_system_dir

def _setup_simulation_directory_resampling(particle, molkey, training_system, resampling_ndx, path_parent_dir, dirname=None):
    if dirname is None:
        dirname = particle.uid
    uid_dir = os.path.join(path_parent_dir, dirname)
    System.mkdir(uid_dir)
    System.rm_dir_content(uid_dir) # Ensure folder is empty 

    molkey_dir = os.path.join(uid_dir, molkey)
    System.mkdir(molkey_dir)
    tr_system_dir = os.path.join(molkey_dir, training_system)
    System.mkdir(tr_system_dir)
    resampling_dir = os.path.join(tr_system_dir, str(resampling_ndx))
    System.mkdir(resampling_dir)
    _generate_itp(particle, molkey, resampling_dir)

    return resampling_dir

def _compute_observables_simulation(particle, molkey, training_system, dirname=None):
    from user.modules.production             import module_production
    from user.modules.production_Tm          import module_production_Tm
    from user.modules.compute_observables    import module_compute_observables
    from user.modules.compute_observables_Tm import module_compute_observables_Tm

    tr_system_dir = _setup_simulation_directory(particle, molkey, training_system, CoreSettings.output_dir, dirname)

    N_retries = 0
    observables_dict = None

    while True:
        try:
            simulation = Sim.Simulation(tr_system_dir, molkey, training_system, particle.current_iteration)

            simulation.add_input_file(os.path.join(tr_system_dir, '%s.itp' %molkey))

            if training_system == 'DPSM256_biphasic':
                simulation.add_module(Sim.Module(module_production_Tm))
                simulation.add_module(Sim.Module(module_compute_observables_Tm))
            else:
                simulation.add_module(Sim.Module(module_production))
                simulation.add_module(Sim.Module(module_compute_observables))

            simulation.run()

            observables_dict = simulation.get_output_variable("observables_dict")

        except System.SimulationError:
            N_retries += 1
            if N_retries > CoreSettings.N_simulation_retry:
                print("Simulation stopped due to an error. Simulation retries exceeded. Exiting.")
                sys.exit(1)
            else:
                print("Simulation stopped due to an error. Retrying simulation. %d retries left." % (CoreSettings.N_simulation_retry - N_retries))
            continue
        break

    return observables_dict

def _compute_observables_simulation_resampling(particle, molkey, training_system, resampling_ndx, dirname=None):
    from user.modules.production             import module_production
    from user.modules.production_Tm          import module_production_Tm
    from user.modules.compute_observables    import module_compute_observables
    from user.modules.compute_observables_Tm import module_compute_observables_Tm

    tr_system_dir = _setup_simulation_directory_resampling(
        particle, molkey, training_system, resampling_ndx, CoreSettings.output_dir, dirname
    )

    N_retries = 0
    observables_dict = None

    while True:
        try:
            simulation = Sim.Simulation(tr_system_dir, molkey, training_system, particle.current_iteration)

            simulation.add_input_file(os.path.join(tr_system_dir, '%s.itp' %molkey))

            if training_system == 'DPSM256_biphasic':
                simulation.add_module(Sim.Module(module_production_Tm))
                simulation.add_module(Sim.Module(module_compute_observables_Tm))
            else:
                simulation.add_module(Sim.Module(module_production))
                simulation.add_module(Sim.Module(module_compute_observables))

            simulation.run()

            observables_dict = simulation.get_output_variable("observables_dict")

        except System.SimulationError:
            N_retries += 1
            if N_retries > CoreSettings.N_simulation_retry:
                print("Simulation stopped due to an error. Simulation retries exceeded. Exiting.")
                sys.exit(1)
            else:
                print("Simulation stopped due to an error. Retrying simulation. %d retries left." % (CoreSettings.N_simulation_retry - N_retries))
            continue
        break

    return observables_dict

def _compute_scores(particle_observables_dict):
    cost = 0
    normalization = ScoringFunctions.get_normalizations(UserSettings.scoring_weights_tr_systems) # normalizes by number of systems each observable is calculated

    for molkey in particle_observables_dict:
        for tr_system in particle_observables_dict[molkey]:
            for observable in UserSettings.scoring_weights_tr_systems[molkey][tr_system]:
                target_value = UserSettings.target_dict[molkey][observable][molkey][tr_system]
                observation = particle_observables_dict[molkey][tr_system][observable]
                sf = UserSettings.scoring_functions_dict[observable]

                cost_i = sf(target_value, observation) / normalization[observable]
                cost_i *= UserSettings.scoring_weights_tr_systems[molkey][tr_system][observable]
                cost_i *= UserSettings.scoring_weights_observables[observable]
                cost += cost_i

    return cost


def _compute_scores_resampling(particle_observables_dict, particle_observables_dict_resampling, resampling_ndx):
    """
    particle observables dict contains the observables from the initial run of a candidate solution.
    particle_observables_dict_resampling contains the observables from one resampling iteration
    """
    cost = 0
    normalization = ScoringFunctions.get_normalizations(UserSettings.scoring_weights_tr_systems) # normalizes by number of systems each observable is calculated

    for molkey in particle_observables_dict:
        for tr_system in particle_observables_dict[molkey]:
            for observable in UserSettings.scoring_weights_tr_systems[molkey][tr_system]:
                target_value = UserSettings.target_dict[molkey][observable][molkey][tr_system]

                if tr_system in UserSettings.resampling_systems[molkey]:
                    observation = particle_observables_dict_resampling[molkey][tr_system][resampling_ndx][observable]
                else:
                    observation = particle_observables_dict[molkey][tr_system][observable]

                sf = UserSettings.scoring_functions_dict[observable]

                cost_i = sf(target_value, observation) / normalization[observable]
                cost_i *= UserSettings.scoring_weights_tr_systems[molkey][tr_system][observable]
                cost_i *= UserSettings.scoring_weights_observables[observable]
                cost += cost_i

    print('total cost: %.3f' %cost)
    return cost





    #################################
    ## PSO setup and settings ##
    #################################

def main():

    restart_file = None #os.path.join(CoreSettings.output_dir, 'CGCompiler-checkpoint-1.pkl.gzip') #None
    rerun_file = None #os.path.join('path/to/', 'rerun.csv')  # rerun can be used to reevaluate a set of candidate solutions for screen to the best procedure
    rerun_restart_file = None # os.path.join(CoreSettings.output_dir, 'rerun-checkpoint-2.pkl.gzip')
    if restart_file is not None:
        o = PSO.PSO.load_checkpoint(restart_file)
        mprint("Restarting from iteration %d" %(o.current_iteration))
        o.run()

    elif rerun_restart_file is not None:
        o = PSO.PSO.load_checkpoint(rerun_restart_file)
        mprint("Restarting from iteration %d" %(o.current_iteration))
        o.rerun()

    elif rerun_file is not None:
        o = PSO.PSO()
        o.description = "Default PSO run"

        o.ptr_observables_function  = _compute_observables_simulation
        o.ptr_scoring_function      = _compute_scores
        o.ptr_observables_function_resampling  = _compute_observables_simulation_resampling
        o.ptr_scoring_function_resampling      = _compute_scores_resampling
        o.population_size           = 2
        o.iterations                = 10
        o.filepath_progress_output  = os.path.join(CoreSettings.output_dir, CoreSettings.name_output_progress)
        o.filepath_gbest_output     = os.path.join(CoreSettings.output_dir, CoreSettings.name_gbest_progress)

        o.initialize_rerun(rerun_file)
        o.rerun()

    else:
        o = PSO.PSO()
        o.description = "Default PSO run"

        o.ptr_observables_function  = _compute_observables_simulation
        o.ptr_scoring_function      = _compute_scores
        o.ptr_observables_function_resampling  = _compute_observables_simulation_resampling
        o.ptr_scoring_function_resampling      = _compute_scores_resampling
        o.population_size           = 4
        o.iterations                = 10
        o.neighborhood_size         = 2            # refers to average neighborhoods, has no effect if average_neighborhood = False in usersettings.py
        o.bResampling               = False #True  # set to True if you want resampling for noise mitigation, which systems to resample is defined in usersettings
        #o.n_pbest_reeval            = 2
        o.n_current_best_reeval     = 2
        o.n_resample_per_particle   = 2
        o.filepath_progress_output  = os.path.join(CoreSettings.output_dir, CoreSettings.name_output_progress)
        o.filepath_gbest_output     = os.path.join(CoreSettings.output_dir, CoreSettings.name_gbest_progress)

        o.initialize()
        o.run()


##########
## MAIN ##
##########

if __name__ == "__main__":
    CoreSettings.parse_user_input()

    main() 