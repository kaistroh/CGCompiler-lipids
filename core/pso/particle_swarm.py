from copy import deepcopy
import os
import sys
import pickle
import user.usersettings as UserSettings
import numpy as np
import pandas as pd
import gzip

from random             import Random
from datetime           import datetime
from mpi4py             import MPI
import core.settings    as CoreSettings
import core.system as   System
from core.system        import mprint

from core.pso.swarm     import Swarm


class PSO:
    @staticmethod
    def __default_observables_function():
        return 0

    @staticmethod
    def __default_scoring_function():
        return 0

    def __init__(self):
        self.global_best            = None

        self.max_iterations         = 1000
        self.iterations             = 1
        self.population_size        = 4
        self.neighborhood_size      = 2
        
        self.filepath_checkpoint        = None
        self.filepath_progress_output   = None
        self.filepath_gbest_output      = None

        self.ptr_observables_function = PSO.__default_observables_function
        self.ptr_scoring_function   = PSO.__default_scoring_function
        self.ptr_observables_function_resampling  = PSO.__default_observables_function
        self.ptr_scoring_function_resampling      = PSO.__default_scoring_function
        self.current_iteration      = 0
        self.start_iteration        = 0
        self.additional_output      = {}

        self.beads_to_optimize      = UserSettings.beads_to_optimize
        self.bonds_to_optimize      = UserSettings.bonds_to_optimize
        self.angles_to_optimize     = UserSettings.angles_to_optimize

        self.feasible_bead_types    = UserSettings.feasible_bead_types
        self.bond_length_min        = UserSettings.bond_length_min
        self.bond_length_max        = UserSettings.bond_length_max
        self.bond_fc_min            = UserSettings.bond_fc_min
        self.bond_fc_max            = UserSettings.bond_fc_max
        self.angle_min              = UserSettings.angle_min
        self.angle_max              = UserSettings.angle_max
        self.angle_fc_min           = UserSettings.angle_fc_min
        self.angle_fc_max           = UserSettings.angle_fc_max
        self.X_cont_upperbounds     = UserSettings.X_cont_upperbounds
        self.X_cont_lowerbounds     = UserSettings.X_cont_lowerbounds


        self.RNG                    = Random()

        self._bRestartRun       = False
        self.description        = "PSO"

        self.bResampling        = False
        self.n_current_best_reeval     = 1
        self.n_resample_per_particle   = 2

    def __check_settings(self):
        def error(name, value, msg):
            mprint("Error:\t" + name + " (" + str(value) + ") " + msg.strip())
            sys.exit(1)

        if self.population_size < 2:
            error("population_size", self.population_size, "should be equal to or larger than 2.")


    def __write_gbest_file(self, current_iteration):
        if MPI is not None: # Ensure only master writes
            if not MPI.COMM_WORLD.Get_rank() == 0:
                return

        def header_string(header):
            return header + ":"
        def body_string(key, value):
            return "\t" + str(key) + "\t" + str(value) + ""
        def footer_string():
            return ""

        def write_to_new_file(data):
            if self.filepath_gbest_output is None:
                return

            with open(self.filepath_gbest_output, 'wt') as file:
                for line in data:
                    print(line)
                    file.write(line + "\n")

        def append_to_file(data):
            if self.filepath_gbest_output is None:
                return
                
            with open(self.filepath_gbest_output, 'at') as file:
                for line in data:
                    file.write(line + "\n")

        def create_progress_file():
            data = []
            data.append("####################")
            data.append("##   CGCompiler   ##")
            data.append("####################")
            data.append("")

            data.append(header_string("Date"))
            data.append(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
            data.append("")

            data.append(header_string("Description"))
            description = self.description.split("\n")
            for line in description:
                data.append("\t" + line)
            data.append("")
            
            data.append(header_string("Settings"))

            data.append(body_string("Iterations:\t\t\t", self.iterations))
            data.append(body_string("Population Size:\t\t", self.population_size))

            # if self.filepath_checkpoint is not None:
            #     data.append(body_string("Filepath checkpoint:\t\t", self.filepath_checkpoint))
            for key, val in self.additional_output.items():
                data.append(body_string(key, val))
            
            data.append(footer_string())

            write_to_new_file(data)

        def write_iteration():
            data        = []

            
            data.append("%d\t%f\t%d\t%s" %(self.current_iteration,
                                           self._population.gbest_fitness,
                                           self._population.gbest_id,
                                           self._population.gbest_uid,
                                           
                                           ))

            append_to_file(data)

        if current_iteration == -1:
            create_progress_file()
        else:
            write_iteration()

                   

    def __write_progress_file(self, current_iteration, output_dict, n_run_dict):
        if MPI is not None: # Ensure only master writes
            if not MPI.COMM_WORLD.Get_rank() == 0:
                return

        def header_string(header):
            return header + ":"
        def body_string(key, value):
            return "\t" + str(key) + "\t" + str(value) + ""
        def footer_string():
            return ""

        def write_to_new_file(data):
            if self.filepath_progress_output is None:
                return

            with open(self.filepath_progress_output, 'wt') as file:
                for line in data:
                    print(line)
                    file.write(line + "\n")

        def append_to_file(data):
            if self.filepath_progress_output is None:
                return
                
            with open(self.filepath_progress_output, 'at') as file:
                for line in data:
                    file.write(line + "\n")


        def create_progress_file():
            data = []
            data.append("####################")
            data.append("##   CGCompiler   ##")
            data.append("####################")
            data.append("")

            data.append(header_string("Date"))
            data.append(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
            data.append("")

            data.append(header_string("Description"))
            description = self.description.split("\n")
            for line in description:
                data.append("\t" + line)
            data.append("")
            
            data.append(header_string("Settings"))

            data.append(body_string("Iterations:\t\t\t", self.iterations))
            data.append(body_string("Population Size:\t\t", self.population_size))

            # if self.filepath_checkpoint is not None:
            #     data.append(body_string("Filepath checkpoint:\t\t", self.filepath_checkpoint))
            for key, val in self.additional_output.items():
                data.append(body_string(key, val))
            
            data.append(footer_string())

            write_to_new_file(data)

        def write_iteration():
            sorted_data = sorted(output_dict.items(), key=lambda x: x[1], reverse=False)
            data        = []

            data.append(header_string("Iteration " + str(current_iteration)))
            data.append(header_string("finished"))
            data.append(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

            for index in sorted_data:
                #data.append(body_string(index[0], str(index[1]) + "\tn_run=" + str(1) ))
                data.append(body_string(index[0], str(index[1])))

            data.append(footer_string())

            append_to_file(data)

        if current_iteration == -1:
            create_progress_file()
        else:
            write_iteration()

    def save_checkpoint(self, filename_prefix='CGCompiler-checkpoint-'):
        if MPI is not None: # Ensure only master writes
            if not MPI.COMM_WORLD.Get_rank() == 0:
                return

        filename = '{0}{1}.pkl.gzip'.format(filename_prefix, self.current_iteration)
        filepath = os.path.join(CoreSettings.output_dir, filename)
        print('Saving checkpoint to {0}'.format(filepath))

        with gzip.open(filepath, 'w', compresslevel=5) as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_checkpoint(filename):
        with gzip.open(filename) as f:
            o = pickle.load(f)
            o._bRestartRun = True
        return o

    def initialize(self, path_restart_file=None):
        """Confirms settings. Prepares algorithm"""

        mprint("Checking settings and initializing the algorithm..")

        self.__check_settings()

        self._population    = Swarm(self)

    def initialize_rerun(self, path_rerun_file):
        self.__check_settings()
        self._population    = Swarm(self)
        params_df = pd.read_csv(path_rerun_file, index_col=[0,1,2], header=[0,1])
        df_reindexed = params_df.reset_index()
        for i, p in enumerate(self._population.population):
            p.from_dfxs( df_reindexed.xs(i) )


        self.rerun_fitness = pd.DataFrame(
            np.zeros(self.population_size),
            dtype='float',
            index=df_reindexed.iloc[:self.population_size]['uid'].values,
            columns=[0]
        )



    def iterate(self):
        self.current_iteration += 1

    def save_RND_state(self):
        if MPI is not None: # Ensure only master writes
            if not MPI.COMM_WORLD.Get_rank() == 0:
                return
 
        if self._bRestartRun:
            filename = 'rndstate_restart.pkl.gzip'
        else:
            filename = 'rndstate.pkl.gzip'

        with gzip.open(filename, 'w', compresslevel=5) as f:
            pickle.dump(self.RNG.getstate(), f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_dataframe(self, filename_prefix='population-'):
        if MPI is not None: # Ensure only master writes
            if not MPI.COMM_WORLD.Get_rank() == 0:
                return

        filename = '{0}{1}.csv'.format(filename_prefix, self.current_iteration)
        filepath = os.path.join(CoreSettings.output_dir, filename)
        print('Saving checkpoint to {0}'.format(filepath))

        df = self._population.to_df(sorted=False)
        df['iteration'] = int(self.current_iteration)
        df.set_index('iteration', append=True, inplace=True)
        df.to_csv(filepath)

    def save_dataframe_resampling(self, filename_prefix='fitness-resampling-'):
        if MPI is not None: # Ensure only master writes
            if not MPI.COMM_WORLD.Get_rank() == 0:
                return

        filename = '{0}{1}.csv'.format(filename_prefix, self.current_iteration)
        filepath = os.path.join(CoreSettings.output_dir, filename)
        print('Saving resampled fitness to {0}'.format(filepath)) 

        df = self._population.fitness_resampling_to_df()
        df['iteration'] = int(self.current_iteration)
        df.set_index('iteration', append=True, inplace=True)
        df.to_csv(filepath)      

    def save_dataframe_rerun(self, filename_prefix='fitness-rerun-'):
        if MPI is not None: # Ensure only master writes
            if not MPI.COMM_WORLD.Get_rank() == 0:
                return

        filename = '{0}{1}.csv'.format(filename_prefix, self.current_iteration)
        filepath = os.path.join(CoreSettings.output_dir, filename)
        print('Saving rerun fitness to {0}'.format(filepath))

        self.rerun_fitness.to_csv(filepath)

    def save_avg_gbest_dataframe(self, filename_prefix='avg_gbest-'):
        if MPI is not None: # Ensure only master writes
            if not MPI.COMM_WORLD.Get_rank() == 0:
                return

        filename = '{0}{1}.csv'.format(filename_prefix, self.current_iteration)
        filepath = os.path.join(CoreSettings.output_dir, filename)
        print('Saving avg gbest to {0}'.format(filepath))

        df = self._population.avg_gbest_to_df()
        df.to_csv(filepath)


    def save_bead_probabilities(self, filename_prefix='bead_probabilities-'):
        if MPI is not None: # Ensure only master writes
            if not MPI.COMM_WORLD.Get_rank() == 0:
                return

        filename = '{0}{1}.json'.format(filename_prefix, self.current_iteration)
        filepath = os.path.join(CoreSettings.output_dir, filename)

        System.write_dict_to_text_file(filepath, self._population.population[0].bead_probabilities)

    def save_params_set(self, filename_prefix='params_set-'):
        if MPI is not None: # Ensure only master writes
            if not MPI.COMM_WORLD.Get_rank() == 0:
                return

        filename = '{0}{1}.json'.format(filename_prefix, self.current_iteration)
        filepath = os.path.join(CoreSettings.output_dir, filename)

        System.write_dict_to_text_file(filepath, self._population.params_sets)

    def cleanup(self):
        # combine dataframe from seperate iterations, delete some things
        pass

    def particle_ids(self):
        ids = []
        for p in self._population.population:
            ids.append(p.id)

        return np.sort(ids)

    def update_rerun_fitness_df(self, fitness_dict):  
        self.rerun_fitness[self.current_iteration] = pd.Series(fitness_dict)

    def update_particle_current_iteration(self):
        for p in self._population.population:
            p.current_iteration = deepcopy(self.current_iteration)


    def run(self):
        mprint("Starting run:\n")

        self.__write_progress_file(-1, None, None)
        self.__write_gbest_file(-1)

        size = 1
        if MPI is not None:
            size = MPI.COMM_WORLD.Get_size()
            if size > 1:
                mprint("Simulations will be distributed over %d MPI threads.\n" % (size))

        # Short fix when running a restart/continuation using a checkpoint file.
        if self._bRestartRun: 
            self.start_iteration = self.current_iteration
            mprint(self.start_iteration)

        for i in range(self.start_iteration, self.iterations):
            self.current_iteration = i
            mprint("Status: Iteration %d/%d." % (
                self.current_iteration, 
                self.iterations,  
                )
            )

            output_observables_dict, output_fitness_dict = self._population.evaluate_fitness()
            #mprint(output_fitness_dict)
            if self.bResampling:
                self._population.sort_by_current_fitness()
                self._population.reevaluate_fitness(output_observables_dict)
                self.save_dataframe_resampling()
                output_fitness_dict = self._population.fitness_averaging(output_fitness_dict)

            mprint(output_fitness_dict)
            self._population.update_pbest()
            self._population.sort_by_fitness()
            if UserSettings.average_neighborhood:
                self._population.update_avg_gbest()
                self.save_avg_gbest_dataframe()
            mprint(self.particle_ids())
            mprint(self._population.to_df(sorted=False))
            self.save_dataframe()
            self.save_bead_probabilities()
            if UserSettings.bAdaptiveParameters:
                self.save_params_set()
            self._population.update_gbest()
            if MPI is not None:
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                if rank == 0:
                    self._population.update_population()
                else:
                    pass
                self._population = comm.bcast(self._population, root=0) 

            else:
                self._population.update_population()

            self.__write_progress_file(i, output_fitness_dict, None)
            self.__write_gbest_file(i)
            mprint("gbest uid: %s\n" %self._population.gbest_uid)
            self.iterate()
            self.save_checkpoint()

            print('##################################################################')

    def rerun(self):
        mprint('Starting reruns:\n')
        self.__write_progress_file(-1, None, None)

        print(self)
        print(self._population.ptr_pso)
        print(self._population.population[0].ptr_swarm.ptr_pso)
        print(self == self._population.population[0].ptr_swarm.ptr_pso)

        size = 1
        if MPI is not None:
            size = MPI.COMM_WORLD.Get_size()
            if size > 1:
                mprint("Simulations will be distributed over %d MPI threads.\n" % (size))

        # Short fix when running a restart/continuation using a checkpoint file.
        if self._bRestartRun: 
            self.start_iteration = self.current_iteration
            mprint(self.start_iteration)

        for i in range(self.start_iteration, self.iterations):
            self.current_iteration = i
            mprint("Status: Iteration %d/%d." % (
                self.current_iteration, 
                self.iterations, 
                )
            )

            output_observables_dict, output_fitness_dict = self._population.evaluate_fitness()
            self.update_rerun_fitness_df(output_fitness_dict)
            self.__write_progress_file(i, output_fitness_dict, None)
            self.save_dataframe_rerun()

            self.current_iteration += 1
            self.update_particle_current_iteration()
            self.save_checkpoint(filename_prefix='rerun-checkpoint-')
        

    