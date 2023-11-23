from mpi4py             import MPI

import user.usersettings as UserSettings
from core.utils         import merge_dicts_deep

class _SimulationBuffer:
    def __init__(self, ptr_observables_function, ptr_observables_function_resampling):
        self.__buffer               = [] #set()
        self.ptr_observables_function   = ptr_observables_function
        self.ptr_observables_function_resampling = ptr_observables_function_resampling

    def __sim_buffer(self, buffer_data):
        output_dict = {}

        for particle, molkey, tr_system in buffer_data:
            print("UID: " + particle.uid)
            particle.compute_observables(output_dict, molkey, tr_system, self.ptr_observables_function)

        return output_dict

    def __sim_buffer_resampling(self, buffer_data):
        output_dict = {}

        for particle, molkey, tr_system, resampling_ndx in buffer_data:
            print("UID: " + particle.uid)
            particle.compute_observables_resampling(output_dict, molkey, tr_system, self.ptr_observables_function_resampling, resampling_ndx)

        return output_dict

    def add(self, particle_set, training_systems):
        for particle in particle_set:
            for molkey in UserSettings.molecule_names:
                for tr_system in training_systems[molkey]:
                    self.__buffer.append( [particle, molkey, tr_system] )

    def add_resampling(self, particle_set, training_systems, n_resampling):
        for particle in particle_set:
            for molkey in UserSettings.molecule_names:
                for tr_system in training_systems[molkey]:
                    for ndx in range(n_resampling):
                        self.__buffer.append( [particle, molkey, tr_system, ndx] )


    def simulate(self): # Returns output dict (seq string, fitness) describing all of the simulated candidate solutions, without references to the object
        state_MPI = False
        if MPI is not None:
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            if size > 1:
                state_MPI = True

        output_dict = None
        
        if state_MPI:
            output_dict = self.__simulate_MPI(comm)
        else:
            output_dict = self.__sim_buffer(self.__buffer)

        self.__buffer = [] #set()
        return output_dict

    def simulate_resampling(self): # Returns output dict (seq string, fitness) describing all of the simulated candidate solutions, without references to the object
        state_MPI = False
        if MPI is not None:
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            if size > 1:
                state_MPI = True

        output_dict = None
        
        if state_MPI:
            output_dict = self.__simulate_MPI_resampling(comm)
        else:
            output_dict = self.__sim_buffer_resampling(self.__buffer)

        self.__buffer = [] #set()
        return output_dict

    # returns an output dict, similar to self.simulate() (i.e. individual __sim_buffer calls return dicts from different threads, combine and return combined dict)
    def __simulate_MPI(self, comm):
        chunk = lambda lst, N_chunks: [lst[i::N_chunks] for i in range(N_chunks)]

        size = comm.Get_size()
        rank = comm.Get_rank()

        number_of_ranks = size
        if number_of_ranks > len(self.__buffer):
            number_of_ranks = len(self.__buffer)

        if rank == 0:   # Master
            # 0. Send #
            list_of_buffers = chunk(list(self.__buffer), number_of_ranks)
            for i in range(1, number_of_ranks):
                comm.send(list_of_buffers[i], dest=i)

            # 2. Simulate own part #
            output_dict = self.__sim_buffer(list_of_buffers[0])
            #print(output_dict)

            # 4. Receive and combine #
            for i in range(1, number_of_ranks):
                slave_dict  = comm.recv(source=i)
                #print(slave_dict)
                #output_dict = {**output_dict, **slave_dict}
                output_dict = merge_dicts_deep(output_dict, slave_dict)
                #print(output_dict)

            # 5. Send to slaves #
            for i in range(1, number_of_ranks):
                comm.send(output_dict, dest=i)

            return output_dict

        else:           # Slave
            # 1. Receive #
            buffer_data     = comm.recv(source=0)

            # 2. Simulate own part #
            output_dict     = self.__sim_buffer(buffer_data)

            # 3. Send results to master #
            comm.send(output_dict, dest=0)

            # 6. Receive combined #
            output_dict     = comm.recv(source=0)

            return output_dict

        #raise RuntimeError("Should not happen.")

    def __simulate_MPI_resampling(self, comm):
        chunk = lambda lst, N_chunks: [lst[i::N_chunks] for i in range(N_chunks)]

        size = comm.Get_size()
        rank = comm.Get_rank()

        number_of_ranks = size
        if number_of_ranks > len(self.__buffer):
            number_of_ranks = len(self.__buffer)

        if rank == 0:   # Master
            # 0. Send #
            list_of_buffers = chunk(list(self.__buffer), number_of_ranks)
            for i in range(1, number_of_ranks):
                comm.send(list_of_buffers[i], dest=i)

            # 2. Simulate own part #
            output_dict = self.__sim_buffer_resampling(list_of_buffers[0])
            #print(output_dict)

            # 4. Receive and combine #
            for i in range(1, number_of_ranks):
                slave_dict  = comm.recv(source=i)
                #print(slave_dict)
                #output_dict = {**output_dict, **slave_dict}
                output_dict = merge_dicts_deep(output_dict, slave_dict)
                #print(output_dict)

            # 5. Send to slaves #
            for i in range(1, number_of_ranks):
                comm.send(output_dict, dest=i)

            return output_dict

        else:           # Slave
            # 1. Receive #
            buffer_data     = comm.recv(source=0)

            # 2. Simulate own part #
            output_dict     = self.__sim_buffer_resampling(buffer_data)

            # 3. Send results to master #
            comm.send(output_dict, dest=0)

            # 6. Receive combined #
            output_dict     = comm.recv(source=0)

            return output_dict

        #raise RuntimeError("Should not happen.")