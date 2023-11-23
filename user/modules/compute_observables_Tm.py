"""
Modules that handle computation of fitness. This file can contain multiple modules.
"""

import os
import numpy as np
import MDAnalysis as mda

import core.settings as CoreSettings
import core.system as System
import core.gromacs as Gromacs
import core.gromacs_commands

import user.usersettings as UserSettings
import user.analysis.observables as Observables


def module_compute_observables_Tm(simulation):

    def path(filename):
        if filename is None:
            return None
        return os.path.join(simulation._temp_dir, filename)  

    class LocalFiles(): # Filenames used within this module

        EDR_dict = {}
        for T in UserSettings.T_list:
            edr_list = []
            for sim_ndx in range(UserSettings.nsim_melt):
                edr_list.append(path(simulation._share.get_filename('prod-%s-%dK-%d-EDR' %(simulation._training_system, T, sim_ndx))))
            EDR_dict[T] = edr_list

        TPR = simulation._share.get_filename("prod-%s-%dK-0-TPR" %(simulation._training_system, UserSettings.T_list[0]))
        XTC = simulation._share.get_filename("prod-%s-%dK-0-XTC" %(simulation._training_system, UserSettings.T_list[0]))

        observables_dict = UserSettings.filename_observables_dict

        prod_EDR_dict = {}
        XVG_dict = {}
        ENERGY_dict = {}
        fitness_dict = {}
        fitness = UserSettings.name_output_score

        stdout_stderr = None
        if CoreSettings.write_stdout_stderr_to_file:
            stdout_stderr = os.path.join(simulation._output_dir, CoreSettings.filename_output_stdout_stderr)
 

    def compute_observables():
        u = mda.Universe(path(LocalFiles.TPR), path(LocalFiles.XTC))
        lipidnames = UserSettings.lipidnames[simulation._training_system]

        Tm, fit, fit_values, apl_averages, apl_errs, apl_threshold, better_model, crit_sig, crit_lin = Observables.melting_temperature_biphasic_apl_wModelSelection(
            EDR_file_dict=LocalFiles.EDR_dict,
            T_list=UserSettings.T_list,
            nsim=UserSettings.nsim_melt,
            u=u,
            lipidnames=lipidnames,
            t_start=UserSettings.t_start_melt,
            t_stop=UserSettings.t_stop_melt,
            bUseWeights=UserSettings.bUseWeightAPLFit,
            T_penalty=UserSettings.T_penalty
        )

        observables_dict = {}
        observables_dict['Tm'] = Tm
        observables_dict['fit_values'] = fit_values
        observables_dict['apl_averages'] = apl_averages
        observables_dict['apl_errs'] = apl_errs
        observables_dict['apl_threshold'] = apl_threshold
        observables_dict['better_model'] = better_model
        observables_dict['crit_sig'] = crit_sig
        observables_dict['crit_lin'] = crit_lin

        return observables_dict        

    observables_dict = compute_observables()
    print(observables_dict)

    System.write_dict_to_text_file(path(LocalFiles.observables_dict), observables_dict)
    simulation._output_variables["observables_dict"] = observables_dict
    simulation._share.add_file("observables_dict_%s" %simulation._training_system,
                               UserSettings.filename_observables_dict, 
                               True
                              )

    if UserSettings.rerun:
        fname_split = (UserSettings.filename_observables_dict).split('.')
        fname_new = fname_split[0] + '_%d.' %simulation._current_iteration + fname_split[1]
        System.write_dict_to_text_file(path(fname_new), observables_dict)
        simulation._share.add_file("observables_dict_%s_%d" %(simulation._training_system, simulation._current_iteration) ,
                                fname_new, 
                                True
                                )