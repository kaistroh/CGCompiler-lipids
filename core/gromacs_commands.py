from importlib.resources import path
import core.settings as CoreSettings
import user.usersettings as UserSettings

def gmx(bQuiet, bNocopyright, bNobackup):
    command = "gmx"
    if bQuiet:
        command += " -quiet"
    if bNocopyright:
        command += " -nocopyright"
    if bNobackup:
        command += " -nobackup"
    return command


def append_gmx_grompp(path_MDP, path_GRO, path_TOP, path_TPR, path_MDOUT, path_NDX=None, path_TOPOUT=None, iMaxwarn=0):
    command =       " grompp"
    command +=      " -f " + path_MDP
    command +=      " -r " + path_GRO
    command +=      " -c " + path_GRO
    command +=      " -p " + path_TOP
    command +=      " -o " + path_TPR
    command +=      " -po " + path_MDOUT
    if path_NDX is not None:
        command +=  " -n " + path_NDX
    if path_TOPOUT is not None:
        command +=  " -pp " + path_TOPOUT
    if iMaxwarn > 0:
        command +=  " -maxwarn " + str(iMaxwarn)
    return command

def append_gmx_mdrun(path_TPR, 
                     path_TRR, 
                     path_GRO, 
                     path_EDR, 
                     path_LOG, 
                     path_CPO, 
                     bVerbose=False, 
                     iN_threads=-1, 
                     iN_MPI_threads=1, 
                     pinstate=0, 
                     path_XTC=None, 
                     path_PULLX=None, 
                     path_PULLF=None, 
                     path_TRR_RERUN=None, 
                     bBondedGPU=False,
                     bUpdateGPU=False,
                     nsteps=-2):
    command =       " mdrun"
    command +=      " -ntmpi " + str(iN_MPI_threads)
    command +=      " -s " + path_TPR
    command +=      " -o " + path_TRR
    if path_XTC is not None:
        command +=  " -x " + path_XTC
    if path_PULLX is not None:
        command += " -px " + path_PULLX
    if path_PULLF is not None:
        command += " -pf " + path_PULLF
    if path_TRR_RERUN is not None:
        command += " -rerun " + path_TRR_RERUN
    command +=      " -c " + path_GRO
    command +=      " -e " + path_EDR
    command +=      " -g " + path_LOG
    command +=      " -cpo " + path_CPO
#    command +=      " -nb cpu " # switch off GPU
    if bVerbose:
        command +=  " -v"
    if iN_threads > 0:
        command +=  " -nt " + str(iN_threads)
    if pinstate == -1:
        command +=  " -pin off"
    elif pinstate == 0:
        command +=  " -pin auto"
    elif pinstate == 1:
        command +=  " -pin on"

    if bBondedGPU:
        command += " -bonded gpu"
    if bUpdateGPU:
        command += " -update gpu"
    command += " -nsteps " + str(nsteps)

    return command

def append_gmx_editconf(path_input_GRO, path_output_GRO, bCenter=False, bBox = False, ndef = False, bPrinc=False, vec3Rotate=(0,0,0), vec3Translate=(0,0,0)):
    command =       " editconf"
    command +=      " -f " + path_input_GRO
    command +=      " -o " + path_output_GRO
    if bPrinc:
        command +=  " -princ"
    if bCenter:
        command +=  " -c"
    if bBox:
        command += " -bt cubic"
        command += " -d 1"
    if ndef:
        command += " -ndef"
    if not vec3Rotate == (0,0,0):
        command +=  " -rotate " + " ".join(str(i) for i in vec3Rotate)
    if not vec3Translate == (0,0,0):
        command +=  " -translate " + " ".join(str(i) for i in vec3Translate)
    return command

def append_gmx_density(
    path_input_XTC,
    path_input_TPR,
    path_input_NDX,
    path_output_XVG,
    axis="z",
    input_center_group=0,
    input_density_group=0,
    iBeginTime=-1,
    bCenter=True,
    nbins=100,
):
    command =       " density"
    command +=      " -f " + path_input_XTC
    command +=      " -s " + path_input_TPR
    command +=      " -n " + path_input_NDX
    command +=      " -o " + path_output_XVG
    command +=      " -d " + axis
    command +=      " -sl " + nbins
    if iBeginTime >= 0:
        command +=      " -b " + str(iBeginTime)
    if bCenter:
        command +=  " -center "
    
    return command

def append_gmx_distance(path_input_XTC, path_input_TPR, path_input_NDX, path_output_XVG, input1, input2, begintime=-1):
    command =       " distance"
    command +=      " -f " + path_input_XTC
    command +=      " -s " + path_input_TPR
    command +=      " -n " + path_input_NDX
    command +=      " -oxyz " + path_output_XVG
    if begintime >= 0:
        command += " -b " + str(begintime)
    command +=      " -select "
    command +=      "\'com of group \"" + str(input1) + "\" plus com of group \"" + str(input2) + "\"\'"
    print(command)
    return command


def append_gmx_energy(path_EDR, path_XVG, iBeginTime=-1):
    command =       " energy"
    command +=      " -f " + path_EDR
    command +=      " -o " + path_XVG
    if iBeginTime >= 0:
        command +=  " -b " + str(iBeginTime)
    return command

def append_gmx_rdf(path_XTC, path_TPR, path_output_XVG, path_NDX=None, iBeginTime=-1, iEndTime=-1, bXY=False, fBin=0.05):
    command          = " rdf"
    command         += " -f " + path_XTC
    command         += " -s " + path_TPR
    if path_NDX:
        command     += " -n " + path_NDX
    command         += " -o " + path_output_XVG
    if bXY:
        command     += " -xy"
    if iBeginTime >= 0:
        command     += " -b " + str(iBeginTime)
    if iEndTime >= 0:
        command     += " -e " + str(iEndTime)
    command         += " -bin " + str(fBin)
    return command

def append_gmx_genion(path_TPR, path_TOP, path_GRO, bNeutral, strPname, strNname):
    command =       " genion"
    command +=      " -s " + path_TPR
    command +=      " -p " + path_TOP
    command +=      " -o " + path_GRO
    if bNeutral:
        command +=  " -neutral"
    command +=      " -pname " + strPname
    command +=      " -nname " + strNname
    return command

def append_gmx_trjconv(path_input_file, path_TPR, path_output_file, bCenter, pbc="mol"):
    command =       " trjconv"
    command +=      " -f " + path_input_file
    command +=      " -s " + path_TPR
    command +=      " -o " + path_output_file
    if bCenter:
        command +=  " -center"
    command +=      " -pbc " + pbc
    return command

# GROMACS Commands #
class GROMACSRunCommands:
    @staticmethod
    def __gmx():
        return gmx(CoreSettings.GMX_QUIET, CoreSettings.GMX_NOCOPYRIGHT, CoreSettings.GMX_NOBACKUP)
    
    @staticmethod
    def GROMPP_NDX(path_MDP, path_GRO, path_TOP, path_TPR, path_MDOUT, path_NDX, max_warnings=0):
        return GROMACSRunCommands.__gmx() + append_gmx_grompp(path_MDP, path_GRO, path_TOP, path_TPR, path_MDOUT, path_NDX=path_NDX, iMaxwarn=max_warnings)

    @staticmethod
    def GROMPP_NDX_TOPOUT(path_MDP, path_GRO, path_TOP, path_TPR, path_MDOUT, path_NDX, path_TOPOUT, max_warnings=0):
        return GROMACSRunCommands.__gmx() + append_gmx_grompp(path_MDP, path_GRO, path_TOP, path_TPR, path_MDOUT, path_NDX=path_NDX, path_TOPOUT=path_TOPOUT, iMaxwarn=max_warnings)
    
    @staticmethod
    def GROMPP(path_MDP, path_GRO, path_TOP, path_TPR, path_MDOUT, max_warnings=0):
        return GROMACSRunCommands.GROMPP_NDX(path_MDP, path_GRO, path_TOP, path_TPR, path_MDOUT, None, max_warnings=max_warnings)

    @staticmethod
    def MDRUN_XTC(path_TPR, path_TRR, path_GRO, path_EDR, path_LOG, path_CPO, path_XTC):
        return GROMACSRunCommands.__gmx() + append_gmx_mdrun(path_TPR, path_TRR, path_GRO, path_EDR, path_LOG, path_CPO, path_XTC=path_XTC, bVerbose=CoreSettings.GMX_MDRUN_VERBOSE, iN_threads=CoreSettings.MDRUN_N_threads, iN_MPI_threads=CoreSettings.GMX_MDRUN_N_MPI_THREADS, pinstate=CoreSettings.GMX_MDRUN_PINSTATE)

    @staticmethod
    def MDRUN_XTC_NSTEPS(path_TPR, path_TRR, path_GRO, path_EDR, path_LOG, path_CPO, path_XTC, nsteps, bBondedGPU=UserSettings.useGPU, bUpdateGPU=UserSettings.updateGPU):
        return GROMACSRunCommands.__gmx() + append_gmx_mdrun(
            path_TPR, 
            path_TRR, 
            path_GRO, 
            path_EDR, 
            path_LOG, 
            path_CPO, 
            path_XTC=path_XTC, 
            bVerbose=CoreSettings.GMX_MDRUN_VERBOSE, 
            iN_threads=CoreSettings.MDRUN_N_threads, 
            iN_MPI_threads=CoreSettings.GMX_MDRUN_N_MPI_THREADS, 
            pinstate=CoreSettings.GMX_MDRUN_PINSTATE,
            nsteps=nsteps,
            bBondedGPU=bBondedGPU,
            bUpdateGPU=bUpdateGPU
        )

    @staticmethod
    def MDRUN_PULL(path_TPR, path_TRR, path_GRO, path_EDR, path_LOG, path_CPO, path_XTC, path_PULLX, path_PULLF):
        return GROMACSRunCommands.__gmx() + append_gmx_mdrun(path_TPR, path_TRR, path_GRO, path_EDR, path_LOG, path_CPO, path_XTC=path_XTC, 
                                                             path_PULLX=path_PULLX, path_PULLF=path_PULLF, 
                                                             bVerbose=CoreSettings.GMX_MDRUN_VERBOSE, iN_threads=CoreSettings.MDRUN_N_threads, iN_MPI_threads=CoreSettings.GMX_MDRUN_N_MPI_THREADS, pinstate=CoreSettings.GMX_MDRUN_PINSTATE,
                                                             bBondedGPU=UserSettings.useGPU)

    @staticmethod
    def MDRUN_PULL_RERUN(path_TPR, path_TRR, path_GRO, path_EDR, path_LOG, path_CPO, path_XTC, path_PULLX, path_PULLF, path_TRR_RERUN):
        return GROMACSRunCommands.__gmx() + append_gmx_mdrun(path_TPR, path_TRR, path_GRO, path_EDR, path_LOG, path_CPO, path_XTC=path_XTC, 
                                                             path_PULLX=path_PULLX, path_PULLF=path_PULLF, path_TRR_RERUN=path_TRR_RERUN,
                                                             bVerbose=CoreSettings.GMX_MDRUN_VERBOSE, iN_threads=CoreSettings.MDRUN_N_threads, iN_MPI_threads=CoreSettings.GMX_MDRUN_N_MPI_THREADS, pinstate=CoreSettings.GMX_MDRUN_PINSTATE,
                                                             bBondedGPU=False)

    @staticmethod
    def MDRUN(path_TPR, path_TRR, path_GRO, path_EDR, path_LOG, path_CPO):
        return GROMACSRunCommands.MDRUN_XTC(path_TPR, path_TRR, path_GRO, path_EDR, path_LOG, path_CPO, None)

    @staticmethod
    def EDITCONF_center(path_input_GRO, path_output_GRO):
        return GROMACSRunCommands.__gmx() + append_gmx_editconf(path_input_GRO, path_output_GRO, bCenter=True)

    @staticmethod
    def EDITCONF_pdb2gro(path_input_GRO, path_output_GRO):
        return GROMACSRunCommands.__gmx() + append_gmx_editconf(path_input_GRO, path_output_GRO, bBox=True)

    @staticmethod
    def EDITCONF_align_X_center(path_input_GRO, path_output_GRO):
        return GROMACSRunCommands.__gmx() + append_gmx_editconf(path_input_GRO, path_output_GRO, bCenter=True, bPrinc=True)

    @staticmethod
    def EDITCONF_align_Z_center(path_input_GRO, path_output_GRO):
        return GROMACSRunCommands.__gmx() + append_gmx_editconf(path_input_GRO, path_output_GRO, bCenter=True, bPrinc=False, vec3Rotate=CoreSettings.GMX_EDITCONF_ALIGN_Z_VEC3ROTATE)

    @staticmethod
    def EDITCONF_rotate(path_input_GRO, path_output_GRO, rotate_vec):
        return GROMACSRunCommands.__gmx() + append_gmx_editconf(path_input_GRO, path_output_GRO, bCenter=False, bPrinc=False, vec3Rotate=rotate_vec)

    @staticmethod
    def EDITCONF_on_top_of_mem(path_input_GRO, path_output_GRO, translate_vec):
        return GROMACSRunCommands.__gmx() + append_gmx_editconf(path_input_GRO, path_output_GRO, bCenter=False, bPrinc=False, vec3Translate=translate_vec)

    @staticmethod
    def ENERGY(path_EDR, path_XVG, iBeginTime):
        return GROMACSRunCommands.__gmx() + append_gmx_energy(path_EDR, path_XVG, iBeginTime=iBeginTime)

    @staticmethod
    def GENION_neutralize(path_TPR, path_TOP, path_GRO, strPname, strNname):
        return GROMACSRunCommands.__gmx() + append_gmx_genion(path_TPR, path_TOP, path_GRO, True, strPname, strNname)

    @staticmethod
    def TRJCONV_center_PBC(path_GRO, path_TPR):
        return GROMACSRunCommands.__gmx() + append_gmx_trjconv(path_GRO, path_TPR, path_GRO, True, pbc="mol")

    @staticmethod
    def DISTANCE(path_XTC, path_TPR, path_NDX, path_output_XVG, input1, input2, begintime):
        return GROMACSRunCommands.__gmx() + append_gmx_distance(path_XTC, path_TPR, path_NDX, path_output_XVG, input1, input2, begintime)

    @staticmethod
    def DENSITY(path_XTC, path_TPR, path_NDX, path_output_XVG):
        return GROMACSRunCommands.__gmx() + append_gmx_density(
            path_input_XTC=path_XTC,
            path_input_TPR=path_TPR,
            path_input_NDX=path_NDX,
            path_output_XVG=path_output_XVG
        )