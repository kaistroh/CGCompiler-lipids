import uuid
from copy import deepcopy
import numpy as np
import pandas as pd

from core.system        import mprint
import core.utils as cutils

import user.usersettings as UserSettings

class Particle(object):

    def __init__(self, X_disc, X_cont, uid, id=None, bead_probabilities=None, ptr_swarm=None):
        self.uid    = uid # unique ID for set each of parameters, needs to be updated as well
        self.id     = id  # fixed id to follow particle through iterations  
        self.ptr_swarm = ptr_swarm
        self.X_disc = X_disc
        self.X_cont = X_cont 

        self.bead_probabilities = deepcopy(bead_probabilities)
        self.bead_probabilities_init = deepcopy(bead_probabilities)

        self.w      = UserSettings.w_init
        self.alpha  = UserSettings.alpha_init
        self.gamma  = UserSettings.gamma
        
        # self.X
        self.V_cont = {}
        for key in self.X_cont:
            self.V_cont[key] = [0.0, 0.0]
            if self.ptr_swarm is not None:
                for j in range(2):
                    self.V_cont[key][j] = self.ptr_swarm.random.uniform(-UserSettings.v_max[key][j], UserSettings.v_max[key][j])

        self.pbest_fitness = None
        self.pbest_fitness_list = []
        self.pbest_disc = deepcopy(X_disc)
        self.pbest_cont = deepcopy(X_cont)
        self.pbest_uid  = deepcopy(uid)  # unique id of that best parameter set
        #self.global_best
        self.p_observables_dict = {}
        self.p_observables_dict_resampling = {}
        self.fitness = None
        self.fitness_dict_resampling = {}
        #self.fitness_list = []
        self.fitness_old = 0.0
        self.fitness_var = None
        self.fitness_se = None


        self.current_iteration = 0
        self.n_current_best_reeval = ptr_swarm.n_current_best_reeval
        self.n_resample_per_particle = ptr_swarm.n_resample_per_particle

    def compute_observables(self, output_dict, molkey, training_system, ptr_observables_function):
        self.p_observables_dict = ptr_observables_function(self, molkey, training_system)
        if self.uid not in output_dict.keys():
            output_dict[self.uid] = {}
        if molkey not in output_dict[self.uid].keys():
            output_dict[self.uid][molkey] = {}
        output_dict[self.uid][molkey][training_system] = self.p_observables_dict

    def compute_observables_resampling(self, output_dict, molkey, training_system, ptr_observables_function_resampling, resampling_ndx):
        self.p_observables_dict_resampling[resampling_ndx] = ptr_observables_function_resampling(self, molkey, training_system, resampling_ndx)
        if self.uid not in output_dict.keys():
            output_dict[self.uid] = {}
        if molkey not in output_dict[self.uid].keys():
            output_dict[self.uid][molkey] = {}
        if training_system not in output_dict[self.uid][molkey].keys():
            output_dict[self.uid][molkey][training_system] = {}
        output_dict[self.uid][molkey][training_system][resampling_ndx] = self.p_observables_dict_resampling[resampling_ndx]

    def compute_fitness(self, observables_dict, ptr_scoring_function):
        # reset some stuff:
        self.fitness_dict_resampling = {}
        self.fitness_var = None
        self.fitness_se = None
        particle_observables_dict = observables_dict[self.uid]
        self.p_observables_dict = deepcopy(particle_observables_dict)
        if self.fitness is not None:
            self.fitness_old = deepcopy(self.fitness)
        self.fitness = ptr_scoring_function(particle_observables_dict)
        self.fitness_dict_resampling['initial'] = deepcopy(self.fitness)

        return self.fitness

    def recompute_fitness(self, initial_observables_dict, resampling_observables_dict, ptr_scoring_function_resampling, resampling_ndx):
        initial_particle_observables_dict = initial_observables_dict[self.uid]
        resampling_particle_observables_dict = resampling_observables_dict[self.uid]
        self.fitness_dict_resampling[resampling_ndx] = ptr_scoring_function_resampling(initial_particle_observables_dict, resampling_particle_observables_dict, resampling_ndx)

        return self.fitness_dict_resampling[resampling_ndx]

    def fitness_resampling_average(self):
        print(self.n_resample_per_particle)
        if len(self.fitness_dict_resampling) == (self.n_resample_per_particle + 1):
            fitness_list = []
            for ndx, fitness in self.fitness_dict_resampling.items():
                fitness_list.append(fitness)
            self.fitness = np.mean(fitness_list)
            self.fitness_var = np.var(fitness_list, ddof=1)
            self.fitness_se = np.std(fitness_list, ddof=1) / np.sqrt(len(fitness_list))
        
        return self.fitness
            

    def update_pbest(self):
        do_update = False
        if self.pbest_fitness is None:
            do_update = True
        else:
            if UserSettings.reverse_sorting:
                if self.fitness < self.pbest_fitness:
                    do_update = True
            else:
                if self.fitness > self.pbest_fitness:
                    do_update = True
        
        if do_update:
            self.pbest_fitness = deepcopy(self.fitness)
            self.pbest_disc = deepcopy(self.X_disc)
            self.pbest_cont = deepcopy(self.X_cont)
            self.pbest_uid  = deepcopy(self.uid)

    def update_bead_probabilities(self, bead_counter, swarm_size):
        for bead, bead_types in self.bead_probabilities.items():
            bt_prob_sum = 0
            for bt, bt_prob in bead_types.items():
                bt_prob_new = self.alpha * bt_prob + (1 - self.alpha) * bead_counter[bead][bt] / swarm_size * 2
                if bt_prob_new < self.bead_probabilities_init[bead][bt] * self.gamma:
                    bt_prob_new = self.bead_probabilities_init[bead][bt] * self.gamma
                self.bead_probabilities[bead][bt] = bt_prob_new
                bt_prob_sum += bt_prob_new
                
        
            ## normalize
            for bt in bead_types:
                self.bead_probabilities[bead][bt] /= bt_prob_sum

        mprint(self.bead_probabilities)


    def update_X_disc(self):
        for bead in self.X_disc:
            bt_probs = list(self.bead_probabilities[bead].values())
            self.X_disc[bead] = self.ptr_swarm.random.choices(self.ptr_swarm.ptr_pso.feasible_bead_types[bead],
                                                             weights=bt_probs)[0] # choices returns a list


    def update_uid(self):
        self.uid = str(uuid.uuid4())

    def update_params(self, w_bar, alpha_bar, rand):
        if rand <= 0.5:
            self.w = cutils.cauchy(w_bar, 0.1)
            self.alpha = cutils.cauchy(alpha_bar, 0.1)
        else:
            self.w = cutils.gaussian(w_bar, 0.1)
            self.alpha = cutils.gaussian(alpha_bar, 0.1)

    def set_w(self, w):
        self.w = deepcopy(w)
        
    def set_alpha(self, alpha):
        self.alpha = deepcopy(alpha)


    def to_df(self):
        index = pd.MultiIndex.from_tuples([(self.id, self.uid)], names=['id', 'uid'])
        col_tuples = []
        data = []
        for key in self.X_cont:
            col_tuples.append((key, 'b0'))
            col_tuples.append((key, 'fc'))
            if UserSettings.bSaveVelocity:
                col_tuples.append((key, 'v_b0'))
                col_tuples.append((key, 'v_fc'))

            data.append(self.X_cont[key][0])
            data.append(self.X_cont[key][1])
            if UserSettings.bSaveVelocity:
                data.append(self.V_cont[key][0])
                data.append(self.V_cont[key][1])
        
        data = np.array(data).reshape(1, -1)
        col_index = pd.MultiIndex.from_tuples(col_tuples)

        
        df = pd.DataFrame(data, index=index, columns=col_index)
        for key in self.X_disc:
            df[key] = self.X_disc[key]

        if self.fitness is not None:
            df['fitness'] = self.fitness

        if UserSettings.bSaveParams:
            df['w'] = self.w
            df['alpha'] = self.alpha 
        
        return df

    def fitness_resampling_to_df(self):
        index_tuples = []
        data = []
        for run_ndx, (key, fitness) in enumerate((self.fitness_dict_resampling.items())):
            index_tuples.append((self.id, self.uid, run_ndx))
            data.append(fitness)
        index = pd.MultiIndex.from_tuples(index_tuples, names=['id', 'uid', 'run_ndx'])

        df = pd.DataFrame(data, index=index, columns=['fitness'])

        return df
        

    def from_dfxs(self, dfxs):
        self.id = dfxs['id'].values[0]
        self.uid = dfxs['uid'].values[0]
        for bead in self.X_disc:
            self.X_disc[bead] = dfxs[bead].values[0]

        for key in self.X_cont:
            self.X_cont[key][0] = dfxs[(key, 'b0')]
            self.X_cont[key][1] = dfxs[(key, 'fc')]
