"""
Modules that handle computation of fitness. This file can contain multiple modules.
"""

import os
import numpy as np
import MDAnalysis as mda
#import panedr

import core.settings as CoreSettings
import core.system as System
import core.gromacs as Gromacs
import core.gromacs_commands

import user.usersettings as UserSettings
import user.analysis.observables as Observables
from user.analysis.rdf import RDF2dCOMperLeaflet

def module_compute_observables(simulation):

    class LocalFiles(): # Filenames used within this module

        EDR = simulation._share.get_filename("prod-%s-EDR" %simulation._training_system)
        TPR = simulation._share.get_filename("prod-%s-TPR" %simulation._training_system)
        XTC = simulation._share.get_filename("prod-%s-XTC" %simulation._training_system)

        observables_dict = UserSettings.filename_observables_dict

        prod_EDR_dict = {}
        XVG_dict = {}
        ENERGY_dict = {}
        fitness_dict = {}
        fitness = UserSettings.name_output_score

        stdout_stderr = None
        if CoreSettings.write_stdout_stderr_to_file:
            stdout_stderr = os.path.join(simulation._output_dir, CoreSettings.filename_output_stdout_stderr)

    def path(filename):
        if filename is None:
            return None
        return os.path.join(simulation._temp_dir, filename)   

    def compute_observables():
        u = mda.Universe(path(LocalFiles.TPR), path(LocalFiles.XTC))
        observables_dict = {}
        lipidnames = UserSettings.lipidnames[simulation._training_system]

        if UserSettings.bCalcObservable[simulation._training_system]['average_area_per_lipid']:
            observables_dict['average_area_per_lipid'] = Observables.average_area_per_lipid(
                u=u,
                edr=path(LocalFiles.EDR), 
                lipidnames=lipidnames
            )
        
        if UserSettings.bCalcObservable[simulation._training_system]['thickness']:
            observables_dict['thickness'] = Observables.membrane_thickness(
                u, 
                UserSettings.selections_thickness[simulation._training_system]
            )

        if UserSettings.bCalcObservable[simulation._training_system]['bond_lengths_av']:
            observables_dict['bond_lengths_av'] = Observables.bond_lengths(
                u=u, 
                molecule_name=simulation._molkey, 
                bonds_to_optimize=UserSettings.bonds_to_optimize[simulation._molkey]
            )[0]

        if UserSettings.bCalcObservable[simulation._training_system]['angles_av']:
            observables_dict['angles_av'] = Observables.angles(
                u=u,
                molecule_name=simulation._molkey,
                angles_to_optimize=UserSettings.angles_to_optimize[simulation._molkey]
            )[0]

        if UserSettings.bCalcObservable[simulation._training_system]['bond_lengths_dist']:
            observables_dict['bond_lengths_dist'] = Observables.bond_lengths(
                u=u, 
                molecule_name=simulation._molkey, 
                bonds_to_optimize=UserSettings.bonds_to_optimize[simulation._molkey]
            )[1]

        if UserSettings.bCalcObservable[simulation._training_system]['angles_dist']:
            observables_dict['angles_dist'] = Observables.angles(
                u=u,
                molecule_name=simulation._molkey,
                angles_to_optimize=UserSettings.angles_to_optimize[simulation._molkey]
            )[1]

        if UserSettings.bCalcObservable[simulation._training_system]['rdf_2d']:
            observables_dict['rdf_2d'] = {
                'DPSM_CHOL': [np.zeros(10),  np.linspace(0, 1, num=10)],
                'DPSM_POPC': [np.zeros(10),  np.linspace(0, 1, num=10)]
                }
            if simulation._training_system == 'POPC_SSM_CHOL':
                print('Calculating RDF')
                dpsm = u.select_atoms('resname DPSM')
                chol = u.select_atoms('resname CHOL')
                popc = u.select_atoms('resname POPC')

                rdf_dpsm_chol = RDF2dCOMperLeaflet(
                    g1=dpsm, 
                    g2=chol, 
                    nbins=30, 
                    range=(0, 15), 
                    virtual_sites=(False, True),
                    g1_selection='resname DPSM', 
                    g2_selection='resname CHOL', 
                    frame_stride=1
                )

                rdf_dpsm_chol.run(start=UserSettings.rdf_start_frame, step=1)
                observables_dict['rdf_2d'] = {
                    'DPSM_CHOL': [rdf_dpsm_chol.results.rdf, rdf_dpsm_chol.results.bins]
                } ## emd_scoring expects [hist, bins]

                rdf_dpsm_popc = RDF2dCOMperLeaflet(
                    g1=dpsm, 
                    g2=popc, 
                    nbins=30, 
                    range=(0, 15), 
                    virtual_sites=(False, False),
                    g1_selection='resname DPSM', 
                    g2_selection='resname POPC', 
                    frame_stride=1
                )

                rdf_dpsm_popc.run(start=UserSettings.rdf_start_frame, step=1)
                observables_dict['rdf_2d']['DPSM_POPC'] = [rdf_dpsm_popc.results.rdf, rdf_dpsm_popc.results.bins]
            ## emd_scoring expects [hist, bins]


        return observables_dict


    observables_dict = compute_observables()

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