import uuid
import numpy as np
import pandas as pd

from operator           import itemgetter
from mpi4py             import MPI
from copy import deepcopy

import user.usersettings as UserSettings

from core.pso.simbuffer import _SimulationBuffer
from core.pso.particle  import Particle
from core.system        import mprint
from core.utils         import my_cauchy

class Swarm(object):

    def __init__(self, ptr_pso):

        self.gbest = None
        self.gbest_fitness = None
        self.gbest_cont = None
        self.gbest_disc = None
        self.gbest_uid  = None
        self.gbest_id   = None
        self.ptr_pso = ptr_pso
        self.population   = [] # set()
        self.population_sorted = []  # sorted from worst to best as in 10.1016/j.swevo.2020.100808
        self.w_bar = UserSettings.w_init
        self.alpha_bar = UserSettings.alpha_init
        self.S = 0
        self.w_set = set()
        self.alpha_set = set()
        self.params_sets = {'w_bar': UserSettings.w_init, 'alpha_bar': UserSettings.alpha_init, 'w': [], 'alpha': []}
        self.random = self.ptr_pso.RNG
        self.simulation_buffer     = _SimulationBuffer(
            self.ptr_pso.ptr_observables_function, 
            self.ptr_pso.ptr_observables_function_resampling
        )

        self.bead_probabilities = None
        if self.ptr_pso.beads_to_optimize['all'] is not None:
            self.bead_probabilities = {}
            for key in self.ptr_pso.beads_to_optimize['all']:
                n_types = len(UserSettings.feasible_bead_types[key])
                self.bead_probabilities[key] = {}
                for bead_type in UserSettings.feasible_bead_types[key]:
                    self.bead_probabilities[key][bead_type] = 1 / n_types

        self.population_size = ptr_pso.population_size
        self.neighborhood_size = ptr_pso.neighborhood_size
        self.n_current_best_reeval = ptr_pso.n_current_best_reeval
        self.n_resample_per_particle = ptr_pso.n_resample_per_particle

        self.avg_gbest_fitness = None
        self.avg_gbest_cont = None
        self.avg_gbest_disc = None

        p_counter = 0
        while len(self.population) < self.population_size:
            self.population.append(self.__generate_particle(p_counter))
            p_counter += 1

        if MPI is not None:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            if rank == 0:
                pass
            else:
                self.population = None
            self.population = comm.bcast(self.population, root=0)


    def __generate_particle(self, id):
        if self.ptr_pso.beads_to_optimize['all'] is not None:
            X_disc = {}
            for bead in self.ptr_pso.beads_to_optimize['all']:
                bead_type = self.random.choice(self.ptr_pso.feasible_bead_types[bead])
                X_disc[bead] = bead_type

        if self.ptr_pso.bonds_to_optimize['all'] is not None and self.ptr_pso.angles_to_optimize['all'] is not None:
            X_cont = {}

        if self.ptr_pso.bonds_to_optimize['all'] is not None:
            for bond in self.ptr_pso.bonds_to_optimize['all']:
                b0_min, fc_min = self.ptr_pso.X_cont_lowerbounds[bond]
                b0_max, fc_max = self.ptr_pso.X_cont_upperbounds[bond]
                b0 = self.random.uniform(b0_min, b0_max)
                b0 = round(b0, UserSettings.n_decimals_bond_length)
                b_fc = self.random.uniform(fc_min, fc_max)
                b_fc = round(b_fc, UserSettings.n_decimals_bond_fc)
                X_cont[bond] = [b0, b_fc]

        if self.ptr_pso.angles_to_optimize['all'] is not None:
            for angle in self.ptr_pso.angles_to_optimize['all']:
                a0_min, a_fc_min = self.ptr_pso.X_cont_lowerbounds[angle]
                a0_max, a_fc_max = self.ptr_pso.X_cont_upperbounds[angle]
                a0 = self.random.uniform(a0_min, a0_max)
                a0 = round(a0, UserSettings.n_decimals_angle)
                a_fc = self.random.uniform(a_fc_min, a_fc_max)
                a_fc = round(a_fc, UserSettings.n_decimals_angle_fc)
                X_cont[angle] = [a0, a_fc]

        p = Particle(X_disc, X_cont, str(uuid.uuid4()), id=id,
                            bead_probabilities=self.bead_probabilities,
                            ptr_swarm=self) 

        return p

    def update_pbest(self):
        for p in self.population:
            p.update_pbest()


    def update_gbest(self):
        pbest_current_generation = self.population_sorted[-1][0] # sorted worst to best
        do_update = False
        if self.gbest_fitness is None:
            do_update = True
        else:
            if UserSettings.reverse_sorting:
                if pbest_current_generation.fitness < self.gbest_fitness:
                    do_update = True
            else:
                if pbest_current_generation.fitness > self.gbest_fitness:
                    do_update = True
        
        if do_update:      
            self.gbest_fitness = deepcopy(pbest_current_generation.fitness)
            self.gbest_disc = deepcopy(pbest_current_generation.pbest_disc)
            self.gbest_cont = deepcopy(pbest_current_generation.pbest_cont)
            self.gbest_uid  = deepcopy(pbest_current_generation.uid)
            self.gbest_id   = deepcopy(pbest_current_generation.id)

    def update_avg_gbest(self):
        particles = self.population_sorted[-self.neighborhood_size:] # sorted by worst to best pbest
        print(particles)
        p0, pbest_fitness = particles[0]
        self.avg_gbest_fitness = deepcopy(pbest_fitness)
        self.avg_gbest_cont = deepcopy(p0.pbest_cont)

        print(self.avg_gbest_fitness)
        print(self.avg_gbest_cont)
        if self.neighborhood_size > 1:
            for p_i, pbest_fitness in particles[1:]:
                self.avg_gbest_fitness += pbest_fitness
                for key in p_i.X_cont:
                    for j in range(2):
                        self.avg_gbest_cont[key][j] += p_i.X_cont[key][j]

            self.avg_gbest_fitness /= self.neighborhood_size
            for key in self.avg_gbest_cont:
                for j in range(2):
                    self.avg_gbest_cont[key][j] /= self.neighborhood_size        

        print(self.avg_gbest_fitness)
        print(self.avg_gbest_cont)
    
    def avg_gbest_to_df(self):
        col_tuples = []
        data = []
        for key in self.avg_gbest_cont:
            col_tuples.append((key, 'b0'))
            col_tuples.append((key, 'fc'))

            data.append(self.avg_gbest_cont[key][0])
            data.append(self.avg_gbest_cont[key][1])

        data = np.array(data).reshape(1, -1)
        col_index = pd.MultiIndex.from_tuples(col_tuples)

        df = pd.DataFrame(data, index=[self.ptr_pso.current_iteration], columns=col_index)

        if self.avg_gbest_fitness is not None:
            df['fitness'] = self.avg_gbest_fitness
        
        return df
        
    def update_population(self):
        if self.ptr_pso.beads_to_optimize['all'] is not None:
            bead_counter = {}
            for key in self.ptr_pso.beads_to_optimize['all']:
                bead_counter[key] = dict(zip(self.ptr_pso.feasible_bead_types[key],
                                             np.zeros(len(self.ptr_pso.feasible_bead_types[key]), dtype='int')))

            for i in range(int(self.population_size / 2), self.population_size):
                p_i = self.population_sorted[i][0]
                for bead, bead_type in p_i.pbest_disc.items():
                    bead_counter[bead][bead_type] += 1


        self.population_old = deepcopy(self.population)
        self.population = [] #set()

        for i in range(self.population_size):
            p_i = self.population_sorted[i][0]
            p_i.update_bead_probabilities(bead_counter, self.population_size)
            p_i.update_X_disc()
            
            for key in p_i.X_cont:
                r = self.random.choice([j for j in range(i, self.population_size)])
                p_r = self.population_sorted[r][0]
                for j in range(2):
                    rand = self.random.uniform(0, 1)
                    V_i_j_new = (p_i.w * p_i.V_cont[key][j]
                                + UserSettings.c * rand * (p_r.pbest_cont[key][j] - p_i.X_cont[key][j]))

                    if UserSettings.average_neighborhood:
                        rand = self.random.uniform(0, 1)
                        V_i_j_new += UserSettings.c_global * rand * (self.avg_gbest_cont[key][j] - p_i.X_cont[key][j])

                    #check max speed
                    if V_i_j_new > np.abs(UserSettings.v_max[key][j]):
                        V_i_j_new = UserSettings.v_max[key][j]
                    elif V_i_j_new < -np.abs(UserSettings.v_max[key][j]):
                        V_i_j_new = -UserSettings.v_max[key][j]
                        
                    p_i.X_cont[key][j] += V_i_j_new
                    # check inside allowed parameters
                    if p_i.X_cont[key][j] > UserSettings.X_cont_upperbounds[key][j]:
                        p_i.X_cont[key][j] = UserSettings.X_cont_upperbounds[key][j]
                        #V_i_j_new *= -1 * UserSettings.boundary_break
                        V_i_j_new *= -1 * self.random.uniform(0, 1)
                    elif p_i.X_cont[key][j] < UserSettings.X_cont_lowerbounds[key][j]:
                        p_i.X_cont[key][j] = UserSettings.X_cont_lowerbounds[key][j]
                        #V_i_j_new *= -1 * UserSettings.boundary_break
                        V_i_j_new *= -1 * self.random.uniform(0, 1)
                    
                    p_i.V_cont[key][j] = V_i_j_new
                    

            p_i.update_uid()
            self.population.append(p_i)
        print('check')

        if UserSettings.bAdaptiveParameters:
            self.update_param_sets()


    def update_param_sets(self):
        print("UPDATING PARAMS")
        print(self.w_set, self.alpha_set)

        for p in self.population:
            do_update = False
            if (p.fitness is None) or (p.fitness_old is None):
                do_update = True
            else:
                if UserSettings.reverse_sorting:
                    if p.fitness < p.fitness_old:
                        do_update = True
                else:
                    if p.fitness > p.fitness_old:
                        do_update = True
            print(p.fitness, p.fitness_old)
            if do_update:
                if UserSettings.bAdaptiveW:
                    self.w_set = self.w_set.union([p.w])
                if UserSettings.bAdaptiveAlpha:
                    self.alpha_set = self.alpha_set.union([p.alpha])
                self.S += 1
                if UserSettings.bAdaptiveW:
                    self.w_bar = ( self.w_bar * self.S + p.w ) / (self.S + 1)
                if UserSettings.bAdaptiveAlpha:
                    self.alpha_bar = ( self.alpha_bar * self.S + p.alpha ) / (self.S + 1)
                
        if UserSettings.bAdaptiveW:
            self.params_sets['w_bar'] = self.w_bar
            self.params_sets['w'] = list(self.w_set)
        if UserSettings.bAdaptiveAlpha:
            self.params_sets['alpha_bar'] = self.alpha_bar 
            self.params_sets['alpha'] = list(self.alpha_set)

        # two loops, calculate new averages with all particles first and then update particles
        print('SETTING PARAMS')
        for p in self.population:
            rand = self.random.uniform(0, 1)
            if rand <= 0.5:
                if UserSettings.bAdaptiveW:
                    w_new = my_cauchy(size=1, x0=self.w_bar, gamma=0.1, RNG=self.random)[0]
                if UserSettings.bAdaptiveAlpha:
                    alpha_new = my_cauchy(size=1, x0=self.alpha_bar, gamma=0.1, RNG=self.random)[0]
            else:
                if UserSettings.bAdaptiveW:
                    w_new = self.random.gauss(self.w_bar, 0.1)
                if UserSettings.bAdaptiveAlpha:
                    alpha_new = self.random.gauss(self.alpha_bar, 0.1)

            
            #p.set_params(w_new, alpha_new)
            if UserSettings.bAdaptiveW:
                p.set_w(w_new)
            if UserSettings.bAdaptiveAlpha:
                p.set_alpha(alpha_new)


    def evaluate_fitness(self):
        # Simulate #
        self.simulation_buffer.add(self.population, UserSettings.training_systems)
        output_observables_dict = self.simulation_buffer.simulate()

        output_fitness_dict = {}
        for particle in self.population:
            output_fitness_dict[particle.uid] = particle.compute_fitness(output_observables_dict, 
                                                                         self.ptr_pso.ptr_scoring_function)

        return output_observables_dict, output_fitness_dict

    def reevaluate_fitness(self, initial_observables_dict):
        reeval_population_tuples = self.population_sorted_current_fitness[-self.n_current_best_reeval:] # sorted by worst to best pbest
        reeval_population = []
        for p, fitness in reeval_population_tuples:
            reeval_population.append(p)

        self.simulation_buffer.add_resampling(reeval_population, UserSettings.resampling_systems, self.ptr_pso.n_resample_per_particle)
        resampling_observables_dict = self.simulation_buffer.simulate_resampling()

        output_fitness_dict = {}
        for particle in reeval_population:
            output_fitness_dict[particle.uid] = {}
            for ndx in range(self.ptr_pso.n_resample_per_particle):
                output_fitness_dict[particle.uid][ndx] = particle.recompute_fitness(
                    initial_observables_dict,
                    resampling_observables_dict, 
                    self.ptr_pso.ptr_scoring_function_resampling,
                    ndx
                )
                                                                   

        return resampling_observables_dict, output_fitness_dict

    def fitness_averaging(self, output_fitness_dict):
        for p in self.population:
            print('before: %.4f' %p.fitness)
            output_fitness_dict[p.uid] = p.fitness_resampling_average()
            print('after: %.4f' %p.fitness)

        return output_fitness_dict
        

    def sort_by_fitness(self):
        
        self.population_sorted = []
        for particle in self.population:
            self.population_sorted.append([particle, particle.pbest_fitness])

        self.population_sorted.sort(key=itemgetter(1), reverse=UserSettings.reverse_sorting)


    def sort_by_current_fitness(self):
        self.population_sorted_current_fitness = []
        for particle in self.population:
            print(particle, particle.fitness)
            self.population_sorted_current_fitness.append([particle, particle.fitness])
        
        self.population_sorted_current_fitness.sort(key=itemgetter(1), reverse=UserSettings.reverse_sorting)

    def to_df(self, sorted=False):
        df_list = []
        if sorted:
            population = self.population_sorted
        else:
            population = self.population
        for p in population:
            df_list.append(p.to_df())

        return pd.concat(df_list)

    def fitness_resampling_to_df(self):
        df_list = []
        for p in self.population:
            df_list.append(p.fitness_resampling_to_df())

        return pd.concat(df_list)
