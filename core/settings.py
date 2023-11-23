import os
import sys
import core.system as System
import getopt

from core.system import mprint

##############
## Settings ##
##############

# Input #
filepath_checkpoint                 = None

# Output #
write_stdout_stderr_to_file         = True
filename_output_stdout_stderr       = "stdout_stderr.log"
output_dir                          = os.path.join(os.getcwd(), "output")
rerun_dir                           = os.path.join(os.getcwd(), "rerun")
name_output_progress                = "output.txt"
name_gbest_progress                 = "gbest.txt"

# Simulation #
MDRUN_N_threads                     = -1 # non-positive -> leave it to GROMACS
N_simulation_retry                  = 1

# Dependency: GROMACS #
GMX_QUIET                           = False
GMX_NOCOPYRIGHT                     = True
GMX_NOBACKUP                        = True
GMX_MDRUN_VERBOSE                   = True
GMX_MDRUN_N_MPI_THREADS             = 1
GMX_MDRUN_PINSTATE                  = -1 # I.E. pin == off  # Usually best to let srun handle thread pinning.
GMX_EDITCONF_ALIGN_Z_VEC3ROTATE     = (0, 90, 0)

###############
## Functions ##
###############

def parse_user_input(): #TODO add help flag
    #NOTE: As we are editing global variables in a function.
    #      Keep in mind when adding new options to the parser.

    global MDRUN_N_threads, output_dir, filepath_checkpoint, write_stdout_stderr_to_file

    try:
        opts, _ = getopt.getopt(sys.argv[1:], "n:h", ["output_dir=", "restart="])
    except getopt.GetoptError as e:
        mprint(e)
        sys.exit(1)

    for opt, arg in opts:
        if  opt in "-n": # Number of threads for MDRUN
            MDRUN_N_threads = int(arg)
        elif opt in "--output_dir":
            output_dir = os.path.realpath(arg)
        # elif opt in "--restart":
        #     filepath_checkpoint = os.path.realpath(arg)

    System.mkdir(output_dir)

def get_filepath_stdout_stderr(directory):
    if write_stdout_stderr_to_file:
        return directory + filename_output_stdout_stderr
    return None

