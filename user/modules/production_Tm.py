"""
MODULE VERSION:
Module that handles production simulations
"""

import os
import copy
import time

import core.settings as CoreSettings
import core.system      as System
import core.gromacs     as Gromacs
import core.gromacs_commands
#from core.gromacstopmodifier import GromacsTopFile

import user.usersettings as UserSettings

def module_production_Tm(simulation):
    for T in UserSettings.T_list:
        for sim_ndx in range(UserSettings.nsim_melt):
            class LocalFiles():


                ITP             = '%s.itp' %simulation._molkey
                ITP_stiff       = '%s.itp' %simulation._molkey

                MDP_NPT_base         = UserSettings.filename_MDP_NPT_Tm_base
                MDP_NPT_equil_base   = UserSettings.filename_MDP_NPT_equil_Tm_base
                MDP_NPT_smalldt_base = UserSettings.filename_MDP_NPT_smalldt_Tm_base

                MDP_NPT         = UserSettings.filename_MDP_NPT_Tm         + '_%dK_%d.mdp'         %(T, sim_ndx)
                MDP_NPT_equil   = UserSettings.filename_MDP_NPT_equil_Tm   + '_%dK_%d.equil.mdp'   %(T, sim_ndx)
                MDP_NPT_smalldt = UserSettings.filename_MDP_NPT_smalldt_Tm + '_%dK_%d.smalldt.mdp' %(T, sim_ndx)
                MDP_EM          = UserSettings.filename_MDP_EM


                GRO_init        = simulation._training_system + '_init.gro'
                TOP             = simulation._training_system + '.top'

                GRO             = UserSettings.name_output_production + '_%dK_%d.gro'          %(T, sim_ndx)
                GRO_equil       = UserSettings.name_output_production + '_%dK_%d.equil.gro'    %(T, sim_ndx)
                GRO_smalldt     = UserSettings.name_output_production + '_%dK_%d.smalldt.gro'  %(T, sim_ndx)
                GRO_EM          = UserSettings.name_output_production + '_%dK_%d.em.gro'       %(T, sim_ndx)

                NDX             = "index.ndx" #UserSettings.name_output_production + '.ndx'

                TPR             = UserSettings.name_output_production + '_%dK_%d.tpr'          %(T, sim_ndx)
                TPR_equil       = UserSettings.name_output_production + '_%dK_%d.equil.tpr'    %(T, sim_ndx)
                TPR_smalldt     = UserSettings.name_output_production + '_%dK_%d.smalldt.tpr'  %(T, sim_ndx)
                TPR_EM          = UserSettings.name_output_production + '_%dK_%d.em.tpr'       %(T, sim_ndx)

                MDOUT           = UserSettings.name_output_production + '_%dK_%d.mdout.mdp'         %(T, sim_ndx)
                MDOUT_equil     = UserSettings.name_output_production + '_%dK_%d.mdout.equil.mdp'   %(T, sim_ndx)
                MDOUT_smalldt   = UserSettings.name_output_production + '_%dK_%d.mdout.smalldt.mdp' %(T, sim_ndx)
                MDOUT_EM        = UserSettings.name_output_production + '_%dK_%d.mdout.em.mdp'      %(T, sim_ndx)

                TRR             = UserSettings.name_output_production + '_%dK_%d.trr'          %(T, sim_ndx)
                TRR_equil       = UserSettings.name_output_production + '_%dK_%d.equil.trr'    %(T, sim_ndx)
                TRR_smalldt     = UserSettings.name_output_production + '_%dK_%d.smalldt.trr'  %(T, sim_ndx)
                TRR_EM          = UserSettings.name_output_production + '_%dK_%d.em.trr'       %(T, sim_ndx)

                EDR             = UserSettings.name_output_production + '_%dK_%d.edr'           %(T, sim_ndx)
                EDR_equil       = UserSettings.name_output_production + '_%dK_%d.equil.edr'     %(T, sim_ndx)
                EDR_smalldt     = UserSettings.name_output_production + '_%dK_%d.smalldt.edr'   %(T, sim_ndx)
                EDR_EM          = UserSettings.name_output_production + '_%dK_%d.em.edr'        %(T, sim_ndx)

                LOG             = UserSettings.name_output_production + '_%dK_%d.log'           %(T, sim_ndx)
                LOG_equil       = UserSettings.name_output_production + '_%dK_%d.equil.log'     %(T, sim_ndx)
                LOG_smalldt     = UserSettings.name_output_production + '_%dK_%d.smalldt.log'   %(T, sim_ndx)
                LOG_EM          = UserSettings.name_output_production + '_%dK_%d.em.log'        %(T, sim_ndx)

                CPT             = UserSettings.name_output_production + '.cpt'
                CPT_equil       = UserSettings.name_output_production + '.equil.cpt'
                CPT_smalldt     = UserSettings.name_output_production + '.smalldt.cpt'
                CPT_EM     = UserSettings.name_output_production + '.em.cpt'

                XTC             = UserSettings.name_output_production + '_%dK_%d.xtc'           %(T, sim_ndx)
                XTC_equil       = UserSettings.name_output_production + '_%dK_%d.equil.xtc'     %(T, sim_ndx)
                XTC_smalldt     = UserSettings.name_output_production + '_%dK_%d.smalldt.xtc'   %(T, sim_ndx)
                XTC_EM          = UserSettings.name_output_production + '_%dK_%d.em.xtc'        %(T, sim_ndx)

                stdout_stderr       = None
                if CoreSettings.write_stdout_stderr_to_file:
                    stdout_stderr = os.path.join(simulation._output_dir, CoreSettings.filename_output_stdout_stderr)

            def path(filename):
                if filename is None:
                    return None
                return os.path.join(simulation._temp_dir, filename)   


            def actual_simulation():
                print("actual simulation")
                # Tm method needs special NDX file
                # print("generating NDX from GRO\n")
                # Gromacs.NDX_from_GRO(
                #     path(LocalFiles.GRO_init),
                #     path(LocalFiles.NDX),
                #     UserSettings.resnm_to_indexgroup_dict
                # )

                print("editing MDPs\n")
                Gromacs.editMDP_Tm_biphasic_equil(
                    filepath_in=path(LocalFiles.MDP_NPT_smalldt_base),
                    filepath_out=path(LocalFiles.MDP_NPT_smalldt),
                    T_solv=T,
                    T_fluid=UserSettings.T_fluid_init,
                    T_gel=UserSettings.T_gel_init,
                    nsteps=UserSettings.nsteps_smalldt
                )

                Gromacs.editMDP_Tm_biphasic_equil(
                    filepath_in=path(LocalFiles.MDP_NPT_equil_base),
                    filepath_out=path(LocalFiles.MDP_NPT_equil),
                    T_solv=T,
                    T_fluid=UserSettings.T_fluid_init,
                    T_gel=UserSettings.T_gel_init,
                    nsteps=UserSettings.nsteps_melt_equil
                )

                Gromacs.editMDP_Tm_biphasic_production(
                    filepath_in=path(LocalFiles.MDP_NPT_base),
                    filepath_out=path(LocalFiles.MDP_NPT),
                    T=T,
                    nsteps=UserSettings.nsteps_melt_prod
                )

                print('Energy minimization\n')
                System.run_command(
                    core.gromacs_commands.GROMACSRunCommands.GROMPP_NDX(
                        path(LocalFiles.MDP_EM),
                        path(LocalFiles.GRO_init),
                        path(LocalFiles.TOP),
                        path(LocalFiles.TPR_EM),
                        path(LocalFiles.MDOUT_EM),
                        path(LocalFiles.NDX),
                    ),
                    tTimeout=UserSettings.timeout_Production_MDRUN,
                    path_stdout=path(LocalFiles.stdout_stderr),
                    path_stderr=path(LocalFiles.stdout_stderr)
                )

                System.run_simulation_command(
                    core.gromacs_commands.GROMACSRunCommands.MDRUN_XTC_NSTEPS(
                        path(LocalFiles.TPR_EM),
                        path(LocalFiles.TRR_EM),
                        path(LocalFiles.GRO_EM),
                        path(LocalFiles.EDR_EM),
                        path(LocalFiles.LOG_EM),
                        path(LocalFiles.CPT_EM),
                        path(LocalFiles.XTC_EM),
                        UserSettings.nsteps_EM,
                        bBondedGPU=False   # EM not compatible with bonded interactions on GPU
                    ),
                    tTimeout=UserSettings.timeout_Production_MDRUN,
                    path_stdout=path(LocalFiles.stdout_stderr),
                    path_stderr=path(LocalFiles.stdout_stderr)
                )

                print("equilibrating with small dt\n")
                System.run_command(
                    core.gromacs_commands.GROMACSRunCommands.GROMPP_NDX(
                        path(LocalFiles.MDP_NPT_smalldt),
                        path(LocalFiles.GRO_EM),
                        path(LocalFiles.TOP),
                        path(LocalFiles.TPR_smalldt),
                        path(LocalFiles.MDOUT_smalldt),
                        path(LocalFiles.NDX),
                    ),
                    tTimeout=UserSettings.timeout_Production_MDRUN,
                    path_stdout=path(LocalFiles.stdout_stderr),
                    path_stderr=path(LocalFiles.stdout_stderr)
                )

                System.run_simulation_command(
                    core.gromacs_commands.GROMACSRunCommands.MDRUN_XTC_NSTEPS(
                        path(LocalFiles.TPR_smalldt),
                        path(LocalFiles.TRR_smalldt),
                        path(LocalFiles.GRO_smalldt),
                        path(LocalFiles.EDR_smalldt),
                        path(LocalFiles.LOG_smalldt),
                        path(LocalFiles.CPT_smalldt),
                        path(LocalFiles.XTC_smalldt),
                        UserSettings.nsteps_smalldt,
                    ),
                    tTimeout=UserSettings.timeout_Production_MDRUN,
                    path_stdout=path(LocalFiles.stdout_stderr),
                    path_stderr=path(LocalFiles.stdout_stderr)
                )


                print("equilibrating with production dt\n")
                System.run_command(
                    core.gromacs_commands.GROMACSRunCommands.GROMPP_NDX(
                        path(LocalFiles.MDP_NPT_equil),
                        path(LocalFiles.GRO_smalldt),
                        path(LocalFiles.TOP),
                        path(LocalFiles.TPR_equil),
                        path(LocalFiles.MDOUT_equil),
                        path(LocalFiles.NDX),
                    ),
                    path_stdout=path(LocalFiles.stdout_stderr),
                    path_stderr=path(LocalFiles.stdout_stderr)
                )

                System.run_simulation_command(
                    core.gromacs_commands.GROMACSRunCommands.MDRUN_XTC_NSTEPS(
                        path(LocalFiles.TPR_equil),
                        path(LocalFiles.TRR_equil),
                        path(LocalFiles.GRO_equil),
                        path(LocalFiles.EDR_equil),
                        path(LocalFiles.LOG_equil),
                        path(LocalFiles.CPT_equil),
                        path(LocalFiles.XTC_equil),
                        UserSettings.nsteps_melt_equil,
                    ),
                    tTimeout=UserSettings.timeout_Production_MDRUN,
                    path_stdout=path(LocalFiles.stdout_stderr),
                    path_stderr=path(LocalFiles.stdout_stderr)
                )



                print("Production\n")
                System.run_command(
                    core.gromacs_commands.GROMACSRunCommands.GROMPP_NDX(
                        path(LocalFiles.MDP_NPT),
                        path(LocalFiles.GRO_equil),
                        path(LocalFiles.TOP),
                        path(LocalFiles.TPR),
                        path(LocalFiles.MDOUT),
                        path(LocalFiles.NDX),
                    ),
                    path_stdout=path(LocalFiles.stdout_stderr),
                    path_stderr=path(LocalFiles.stdout_stderr)
                )

                System.run_simulation_command(
                    core.gromacs_commands.GROMACSRunCommands.MDRUN_XTC_NSTEPS(
                        path(LocalFiles.TPR),
                        path(LocalFiles.TRR),
                        path(LocalFiles.GRO),
                        path(LocalFiles.EDR),
                        path(LocalFiles.LOG),
                        path(LocalFiles.CPT),
                        path(LocalFiles.XTC),
                        UserSettings.nsteps_melt_prod,
                    ),
                    tTimeout=UserSettings.timeout_Production_MDRUN,
                    path_stdout=path(LocalFiles.stdout_stderr),
                    path_stderr=path(LocalFiles.stdout_stderr)
                )



            System.copy_dir(UserSettings.path_basedir_production + simulation._training_system, path(""))

            actual_simulation()
            
            simulation._share.add_file('smalldt-%s-%dK-%d-GRO'   %(simulation._training_system, T, sim_ndx), LocalFiles.GRO_smalldt, UserSettings.SAVE_EQUIL)
            simulation._share.add_file('smalldt-%s-%dK-%d-TPR'   %(simulation._training_system, T, sim_ndx), LocalFiles.TPR_smalldt, UserSettings.SAVE_EQUIL)
            simulation._share.add_file('smalldt-%s-%dK-%d-XTC'   %(simulation._training_system, T, sim_ndx), LocalFiles.XTC_smalldt, UserSettings.SAVE_EQUIL)
            simulation._share.add_file('smalldt-%s-%dK-%d-TRR'   %(simulation._training_system, T, sim_ndx), LocalFiles.TRR_smalldt, False)
            simulation._share.add_file('smalldt-%s-%dK-%d-EDR'   %(simulation._training_system, T, sim_ndx), LocalFiles.EDR_smalldt, UserSettings.SAVE_EQUIL)
            simulation._share.add_file('smalldt-%s-%dK-%d-LOG'   %(simulation._training_system, T, sim_ndx), LocalFiles.LOG_smalldt, UserSettings.SAVE_EQUIL)
            simulation._share.add_file('smalldt-%s-%dK-%d-MDOUT' %(simulation._training_system, T, sim_ndx), LocalFiles.MDOUT_smalldt, UserSettings.SAVE_EQUIL)
            simulation._share.add_file('smalldt-%s-%dK-%d-MDP'   %(simulation._training_system, T, sim_ndx), LocalFiles.MDP_NPT_smalldt, UserSettings.SAVE_EQUIL)


            simulation._share.add_file('equil-%s-%dK-%d-GRO'   %(simulation._training_system, T, sim_ndx), LocalFiles.GRO_equil, UserSettings.SAVE_EQUIL)
            simulation._share.add_file('equil-%s-%dK-%d-TPR'   %(simulation._training_system, T, sim_ndx), LocalFiles.TPR_equil, UserSettings.SAVE_EQUIL)
            simulation._share.add_file('equil-%s-%dK-%d-XTC'   %(simulation._training_system, T, sim_ndx), LocalFiles.XTC_equil, UserSettings.SAVE_EQUIL)
            simulation._share.add_file('equil-%s-%dK-%d-TRR'   %(simulation._training_system, T, sim_ndx), LocalFiles.TRR_equil, False)
            simulation._share.add_file('equil-%s-%dK-%d-EDR'   %(simulation._training_system, T, sim_ndx), LocalFiles.EDR_equil, UserSettings.SAVE_EQUIL)
            simulation._share.add_file('equil-%s-%dK-%d-LOG'   %(simulation._training_system, T, sim_ndx), LocalFiles.LOG_equil, UserSettings.SAVE_EQUIL)
            simulation._share.add_file('equil-%s-%dK-%d-MDOUT' %(simulation._training_system, T, sim_ndx), LocalFiles.MDOUT_equil, UserSettings.SAVE_EQUIL)
            simulation._share.add_file('equil-%s-%dK-%d-MDP'   %(simulation._training_system, T, sim_ndx), LocalFiles.MDP_NPT_equil, UserSettings.SAVE_EQUIL)


            simulation._share.add_file('prod-%s-%dK-%d-GRO'   %(simulation._training_system, T, sim_ndx), LocalFiles.GRO, UserSettings.WRITE_GRO)
            #simulation._share.add_file('prod-%s-NDX'          %(simulation._training_system),             LocalFiles.NDX, True)
            simulation._share.add_file('prod-%s-%dK-%d-TPR'   %(simulation._training_system, T, sim_ndx), LocalFiles.TPR, UserSettings.WRITE_TPR)
            simulation._share.add_file('prod-%s-%dK-%d-XTC'   %(simulation._training_system, T, sim_ndx), LocalFiles.XTC, UserSettings.WRITE_XTC)
            simulation._share.add_file('prod-%s-%dK-%d-TRR'   %(simulation._training_system, T, sim_ndx), LocalFiles.TRR, UserSettings.WRITE_TRR)
            simulation._share.add_file('prod-%s-%dK-%d-EDR'   %(simulation._training_system, T, sim_ndx), LocalFiles.EDR, UserSettings.WRITE_EDR)
            simulation._share.add_file('prod-%s-%dK-%d-LOG'   %(simulation._training_system, T, sim_ndx), LocalFiles.LOG, UserSettings.WRITE_LOG)
            simulation._share.add_file('prod-%s-%dK-%d-MDOUT' %(simulation._training_system, T, sim_ndx), LocalFiles.MDOUT, False)
            simulation._share.add_file('prod-%s-%dK-%d-MDP'   %(simulation._training_system, T, sim_ndx), LocalFiles.MDP_NPT, UserSettings.SAVE_EQUIL)