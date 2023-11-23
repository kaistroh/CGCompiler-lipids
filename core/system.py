"""Miscellaneous IO and other system functions."""

import os
import sys
import shutil
import subprocess
import signal
import json
from distutils.dir_util import copy_tree

from core.utils import NumpyEncoder

from mpi4py import MPI
def mprint(string, end="\n"):
    if MPI is not None:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(string,end=end)
    else:
        print(string,end=end)

class _Process:
    """Keeps track of spawned processes. Handles cleanup of processes on program termination."""
    _active_processes = set()

    @classmethod
    def _new_process(cls, process):
        cls._active_processes.add(process)

        return process

    @classmethod
    def run(cls, command, stdout, stderr, timeout=None, input=None, cwd=None):
        returncode = -1

        process = cls._new_process(subprocess.Popen(command.split(), cwd=cwd, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr))
        try:
            process.communicate(input=input, timeout=timeout)
        except subprocess.TimeoutExpired:
            print("Timeout expired for process id:\t%d. Killing process." % (process.pid))
            process.kill()
            returncode = 2
        else:
            returncode = process.returncode
        cls._active_processes.discard(process)

        return returncode


    @staticmethod
    def _getsig(signum):
        return signal.Signals(signum).name

    @classmethod
    def terminate(cls, signum, frame):
        mprint("\nSignal caught:\t%s\nTerminating simulations...\n" % (cls._getsig(signum)))
        for p in cls._active_processes:
            p.kill()
            print("killed simulation with process id:\t%d" % (p.pid))

        sys.exit(1)
    
    @classmethod
    def register(cls):
        signal.signal(signal.SIGINT, cls.terminate)
        signal.signal(signal.SIGTERM, cls.terminate)

_Process.register()

######################
## COMMON FUNCTIONS ##
######################

def replace_in_file(filepath, tag, replacement):
    """
    Replaces strings 'tag' with string 'replacement' in file 'filepath'.
    """
    data = []
    with open(filepath, 'rt') as file:
        for row in file:
            if tag in row:
                row = row.replace(tag,replacement)
            data.append(row)

    os.remove(filepath)
    
    with open(filepath, 'wt') as file:
        for row in data:
            file.write(row)

def read_text_file(filepath, strip=False):
    data = []
    if strip:
        with open(filepath, 'rt') as file:
            for line in file:
                data.append(line.strip())
    else:
        with open(filepath, 'rt') as file:
            data = file.readlines()
    return data

def write_text_file(filepath, data, add_newline=False):
    if add_newline:
        with open(filepath, 'wt') as file:
            for row in data:
                file.write(row + "\n")
    else:
        with open(filepath, 'wt') as file:
            for row in data:
                file.write(row)

def write_dict_to_text_file(filepath, data):
    with open(filepath, 'wt') as file:
        file.write(json.dumps(data, cls=NumpyEncoder))

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

def get_file_dir(file):
    return os.path.dirname(os.path.realpath(file))

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def rm_dir_content(path_to_dir):
    for filename in os.listdir(path_to_dir):
        file_path = os.path.join(path_to_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def copy_dir_files(src_dir, dest_dir):
    list_files = os.listdir(src_dir)
    for filename in list_files:
        filepath = os.path.join(src_dir, filename)
        if os.path.isfile(filepath):
            shutil.copy(filepath, dest_dir)

def copy_dir(src_dir, dest_dir):
    copy_tree(src_dir, dest_dir)


def check_exitcode(exitcode):
    if not exitcode == 0:
        sys.exit(exitcode)

################
## SUBPROCESS ##
################

def _try_open(filepath, mode):
    if filepath is not None:
        mgr = open(filepath, mode)
    else:
        mgr = open(os.devnull, "w")
    return mgr

def run_command(command, tTimeout=None, input=None, path_stdout=None, path_stderr=None, cwd=None):
    hdlOut = _try_open(path_stdout, 'at')
    hdlErr = _try_open(path_stderr, 'at')

    with hdlOut as out, hdlErr as err:
        returncode = _Process.run(command, stdout=out, stderr=err, timeout=tTimeout, input=input, cwd=cwd)

    check_exitcode(returncode)


class SimulationError(Exception):
    """ Should be raised when a GROMACS MDRUN call exits with an error."""
    pass

def _sim_check_exitcode(exitcode):
    if not exitcode == 0:
        raise SimulationError

def run_simulation_command(command, tTimeout=None, path_stdout=None, path_stderr=None):
    hdlOut = _try_open(path_stdout, 'at')
    hdlErr = _try_open(path_stderr, 'at')

    with hdlOut as out, hdlErr as err:
        returncode = _Process.run(command, stdout=out, stderr=err, timeout=tTimeout)

    _sim_check_exitcode(returncode)

