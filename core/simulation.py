from tempfile import TemporaryDirectory
import shutil
import os
import sys

class Module():
    def __init__(self, module_pointer):
        self._ptr = module_pointer  # ptr to module function

    def run(self, simulation):
        return self._ptr(simulation)

class Simulation():
    class SharedFiles():  # Contains filenames in the temporary directory that need to be shared between modules/output folder.
        class File():
            def __init__(self, filename, bWrite):
                self.filename = filename
                self.bWrite = bWrite

        def __init__(self, src_dir):
            self.dir = src_dir
            self._files = dict()

        def add_file(self, str_ID, filename, bWrite):
            if str_ID not in self._files:
                self._files[str_ID] = self.File(filename, bWrite)
            else:
                raise ValueError("str_ID already present in the SharedFiles object. Check modules for overlapping str_ID's.")

        def get_filename(self, str_ID):
            return self._files[str_ID].filename

        def write_to_output_dir(self, dst_dir):
            for file_obj in self._files.values():
                if file_obj.bWrite:
                    file_obj.bWrite = False
                    shutil.copyfile(os.path.join(self.dir, file_obj.filename), os.path.join(dst_dir, file_obj.filename))

    def __init__(self, output_dir, molkey, training_system, current_iteration=0):
        self._molkey            = molkey
        self._training_system   = training_system # name of training system, should be identical to folder name of that system
        self._input_data_paths  = []            # List of paths to input files, will be copied to temp dir upon start of run()
        self._output_dir        = output_dir    # To which certain output (specified by the called module) will be copied from the temp_dir
        self._modules           = []            # Contains module objects in the order of execution
        self._output_variables  = dict()        # Contains variable output from modules

        self._share             = None          # SharedFiles object accessed by the modules
        self._temp_dir          = None          # In which all simulation input/output will be stored, created/destroyed outside the class

        self._current_iteration = current_iteration

    def add_input_file(self, filepath):
        if os.path.isfile(filepath):
            self._input_data_paths.append(filepath)
        else:
            print("ERROR: filepath " + str(filepath) + " does not point to a file.\nExiting.\n")
            sys.exit(1)

    def add_module(self, module_pointer):
        self._modules.append(module_pointer)

    def run(self):
        with TemporaryDirectory() as self._temp_dir:
            for path in self._input_data_paths:
                filename = os.path.basename(path)
                shutil.copyfile(path, os.path.join(self._temp_dir, filename))

            self._share = self.SharedFiles(self._temp_dir)
            for module in self._modules:
                module.run(self)
                self._share.write_to_output_dir(self._output_dir)

    def get_output_variable(self, key):
        return self._output_variables[key]

