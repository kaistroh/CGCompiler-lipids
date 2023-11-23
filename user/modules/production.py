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

def module_production(simulation):

    class LocalFiles():

        ITP             = '%s.itp' %simulation._molkey

        MDP_NPT         = UserSettings.filename_MDP_NPT
        MDP_NPT_equil   = UserSettings.filename_MDP_NPT_equil
        MDP_NPT_smalldt = UserSettings.filename_MDP_NPT_smalldt
        MDP_EM          = UserSettings.filename_MDP_EM

        GRO_init        = simulation._training_system + '_init.gro'
        TOP             = simulation._training_system + '.top'

        GRO             = UserSettings.name_output_production + '.gro'
        GRO_equil       = UserSettings.name_output_production + '.equil.gro'
        GRO_smalldt     = UserSettings.name_output_production + '.smalldt.gro'
        GRO_EM          = UserSettings.name_output_production + '.em.gro'

        NDX             = UserSettings.name_output_production + '.ndx'

        TPR             = UserSettings.name_output_production + '.tpr'
        TPR_equil       = UserSettings.name_output_production + '.equil.tpr'
        TPR_smalldt     = UserSettings.name_output_production + '.smalldt.tpr'
        TPR_EM          = UserSettings.name_output_production + '.em.tpr'

        MDOUT           = UserSettings.name_output_production + '.mdout.mdp'
        MDOUT_equil     = UserSettings.name_output_production + '.mdout.equil.mdp'
        MDOUT_smalldt   = UserSettings.name_output_production + '.mdout.smalldt.mdp'
        MDOUT_EM        = UserSettings.name_output_production + '.mdout.em.mdp'

        TRR             = UserSettings.name_output_production + '.trr'
        TRR_equil       = UserSettings.name_output_production + '.equil.trr'
        TRR_smalldt     = UserSettings.name_output_production + '.smalldt.trr'
        TRR_EM          = UserSettings.name_output_production + '.em.trr'

        EDR             = UserSettings.name_output_production + '.edr'
        EDR_equil       = UserSettings.name_output_production + '.equil.edr'
        EDR_smalldt     = UserSettings.name_output_production + '.smalldt.edr'
        EDR_EM          = UserSettings.name_output_production + '.em.edr'

        LOG             = UserSettings.name_output_production + '.log'
        LOG_equil       = UserSettings.name_output_production + '.equil.log'
        LOG_smalldt     = UserSettings.name_output_production + '.smalldt.log'
        LOG_EM          = UserSettings.name_output_production + '.em.log'

        CPT             = UserSettings.name_output_production + '.cpt'
        CPT_equil       = UserSettings.name_output_production + '.equil.cpt'
        CPT_smalldt     = UserSettings.name_output_production + '.smalldt.cpt'
        CPT_EM          = UserSettings.name_output_production + '.em.cpt'

        XTC             = UserSettings.name_output_production + '.xtc'
        XTC_equil       = UserSettings.name_output_production + '.equil.xtc'
        XTC_smalldt     = UserSettings.name_output_production + '.smalldt.xtc'
        XTC_EM          = UserSettings.name_output_production + '.em.xtc'

        stdout_stderr       = None
        if CoreSettings.write_stdout_stderr_to_file:
            stdout_stderr = os.path.join(simulation._output_dir, CoreSettings.filename_output_stdout_stderr)

    def path(filename):
        if filename is None:
            return None
        return os.path.join(simulation._temp_dir, filename)   


    def actual_simulation():
        print("actual simulation")
        print("generating NDX from GRO\n")
        Gromacs.NDX_from_GRO(
            path(LocalFiles.GRO_init),
            path(LocalFiles.NDX),
            UserSettings.resnm_to_indexgroup_dict
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
                UserSettings.nsteps_equil,
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
                UserSettings.nsteps_dict[simulation._training_system],
            ),
            tTimeout=UserSettings.timeout_Production_MDRUN,
            path_stdout=path(LocalFiles.stdout_stderr),
            path_stderr=path(LocalFiles.stdout_stderr)
        )



    System.copy_dir(UserSettings.path_basedir_production + simulation._training_system, path(""))

    actual_simulation()
    
    simulation._share.add_file('smalldt-%s-GRO' %simulation._training_system, LocalFiles.GRO_smalldt, UserSettings.SAVE_EQUIL)
    simulation._share.add_file('smalldt-%s-TPR' %simulation._training_system, LocalFiles.TPR_smalldt, UserSettings.SAVE_EQUIL)
    simulation._share.add_file('smalldt-%s-XTC' %simulation._training_system, LocalFiles.XTC_smalldt, UserSettings.SAVE_EQUIL)
    simulation._share.add_file('smalldt-%s-TRR' %simulation._training_system, LocalFiles.TRR_smalldt, UserSettings.SAVE_EQUIL)
    simulation._share.add_file('smalldt-%s-EDR' %simulation._training_system, LocalFiles.EDR_smalldt, UserSettings.SAVE_EQUIL)
    simulation._share.add_file('smalldt-%s-LOG' %simulation._training_system, LocalFiles.LOG_smalldt, UserSettings.SAVE_EQUIL)

    simulation._share.add_file('equil-%s-GRO' %simulation._training_system, LocalFiles.GRO_equil, UserSettings.SAVE_EQUIL)
    simulation._share.add_file('equil-%s-TPR' %simulation._training_system, LocalFiles.TPR_equil, UserSettings.SAVE_EQUIL)
    simulation._share.add_file('equil-%s-XTC' %simulation._training_system, LocalFiles.XTC_equil, UserSettings.SAVE_EQUIL)
    simulation._share.add_file('equil-%s-TRR' %simulation._training_system, LocalFiles.TRR_equil, UserSettings.SAVE_EQUIL)
    simulation._share.add_file('equil-%s-EDR' %simulation._training_system, LocalFiles.EDR_equil, UserSettings.SAVE_EQUIL)
    simulation._share.add_file('equil-%s-LOG' %simulation._training_system, LocalFiles.LOG_equil, UserSettings.SAVE_EQUIL)

    simulation._share.add_file('prod-%s-GRO' %simulation._training_system, LocalFiles.GRO, UserSettings.WRITE_GRO)
    simulation._share.add_file('prod-%s-NDX' %simulation._training_system, LocalFiles.NDX, UserSettings.WRITE_NDX)
    simulation._share.add_file('prod-%s-TPR' %simulation._training_system, LocalFiles.TPR, UserSettings.WRITE_TPR)
    simulation._share.add_file('prod-%s-XTC' %simulation._training_system, LocalFiles.XTC, UserSettings.WRITE_XTC)
    simulation._share.add_file('prod-%s-TRR' %simulation._training_system, LocalFiles.TRR, UserSettings.WRITE_TRR)
    simulation._share.add_file('prod-%s-EDR' %simulation._training_system, LocalFiles.EDR, UserSettings.WRITE_EDR)
    simulation._share.add_file('prod-%s-LOG' %simulation._training_system, LocalFiles.LOG, UserSettings.WRITE_LOG)

