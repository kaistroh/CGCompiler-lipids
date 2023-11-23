import os
import sys
import json
from functools import partial
import numpy as np
import core.system as System
from core.gromacstopmodifier import GromacsTopFile
from core.utils import check_observables_calc

import user.analysis.scoring_functions as ScoringFunctions
from user.analysis.utils import read_distribution_file

##############
## Settings ##
##############
rerun                   = False #True
average_neighborhood    = False #True # Still implemented, but did not improve results


path_basedir_production             = System.get_file_dir(__file__) + "/base/production/"

name_output_score                   = "fitness"

training_systems        = {
        'DPSM': [
                'DPSM128_328K',
                #'DPSM256_biphasic',
                'POPC_SSM_CHOL',
   
        ],

}
resampling_systems        = {
        'DPSM': [
                #'DPSM128_328K',
                'DPSM256_biphasic',
                #'POPC_SSM_CHOL',
        ], 

}

opt_headgroup           = False
opt_linker              = True
opt_tails               = False

##### lipid to optimize

baseITPs = {
        'DPSM': GromacsTopFile(System.get_file_dir(__file__) + '/data/martini_v3.0_DPSM.itp', itp=True),
}

molecule_names = [
        "DPSM", 
]

beads = {
        'DPSM': {
                "NC3":  1, 
                "PO4":  2, 
                "AM1":  3, 
                "AM2":  4, 
                "T1A":  5, 
                "C2A":  6, 
                "C3A":  7, 
                "C1B":  8, 
                "C2B":  9, 
                "C3B": 10, 
                "C4B": 11
        },
}

bonds = {
        'DPSM':
                [
                        "NC3-PO4", 
                        "PO4-AM1", 
                        "AM1-AM2", 
                        "AM1-T1A", 
                        "T1A-C2A", 
                        "C2A-C3A", 
                        "AM2-C1B", 
                        "C1B-C2B", 
                        "C2B-C3B", 
                        "C3B-C4B"],
}

angles = {
        'DPSM': 
                [
                "PO4-AM1-AM2", # 2 3 4
                "PO4-AM1-T1A", # 2 3 5
                "AM1-T1A-C2A", # 3 5 6
                "T1A-C2A-C3A", # 5 6 7
                "AM2-C1B-C2B", # 4 8 9
                "C1B-C2B-C3B", # 8 9 10
                "C2B-C3B-C4B"  # 9 10 11
                ],
}    

beads_head = {
        'DPSM': ["NC3", "PO4"],
}

beads_linker    = {
        'DPSM': ["AM1", "AM2"],
}

beads_tails     = {
        'DPSM': ["T1A", "C2A", "C3A", "C1B", "C2B", "C3B", "C4B"],
}

beads_to_optimize = {
        'all':  ['AM1', 'AM2'],
        'DPSM': ["AM1", "AM2"],
        #'DPCE': ["AM1", "AM2"]
}

bonds_to_optimize = {
        'all':  ["PO4-AM1", "AM1-AM2", "AM1-T1A", "AM2-C1B"],
        'DPSM': ["PO4-AM1", "AM1-AM2", "AM1-T1A", "AM2-C1B"],
        #'DPCE': ["AM1-AM2", "AM1-T1A", "AM2-C1B"]  
}

angles_to_optimize = {
        'all': ["PO4-AM1-AM2",                              
                 "PO4-AM1-T1A",
                 "AM1-T1A-C2A",
                 #"T1A-C2A-C3A",
                 "AM2-C1B-C2B"
                ],
        'DPSM': ["PO4-AM1-AM2",                              
                 "PO4-AM1-T1A",
                 "AM1-T1A-C2A",
                 #"T1A-C2A-C3A",
                 "AM2-C1B-C2B"
                ],
        # 'DPCE': ["AM1-T1A-C2A",
        #          #"T1A-C2A-C3A",
        #          "AM2-C1B-C2B"
        #         ]
}

#### set of bead types to choose from, select for each bead individually

tail_bead_types = ["C1", "C2", "C3", "C4", "C5", "C6"]

feasible_bead_types = {"NC4": ["Q1", "Q2", "Q3", "Q4", "Q5"],
                       "PO4": ["Q1", "Q2", "Q3", "Q4", "Q5"],
                       "AM1": ["P1", "P2", "P3", "P4", "P5", "P6",
                               "SP1", "SP2", "SP3", "SP4", "SP5", "SP6",
                               ],
                       "AM2": ["P1", "P2", "P3", "P4", "P5", "P6",
                               "SP1", "SP2", "SP3", "SP4", "SP5", "SP6",
                               ],
                       "T1A": tail_bead_types,
                       "C2A": tail_bead_types,
                       "C3A": tail_bead_types,
                       "C1B": tail_bead_types,
                       "C2B": tail_bead_types,
                       "C3B": tail_bead_types,
                       "C4B": tail_bead_types
                       }

### bonded parameter settings

bond_length_min = 0.30
bond_length_max = 0.55
n_decimals_bond_length = 4

bond_fc_min = 1000
bond_fc_max = 9000
n_decimals_bond_fc = 1

angle_min = 90
angle_max = 180

angle_tail_min = 180
angle_tail_max = 180
n_decimals_angle = 2

angle_fc_min = 5
angle_fc_max = 150 #100
n_decimals_angle_fc = 1


X_cont_upperbounds = {"PO4-AM1": [0.40, bond_fc_max], 
                      "AM1-AM2": [0.35, bond_fc_max], 
                      "AM1-T1A": [0.55, bond_fc_max], 
                      "AM2-C1B": [0.50, bond_fc_max],
                      "PO4-AM1-AM2": [angle_max, angle_fc_max],                              
                      "PO4-AM1-T1A": [angle_max, angle_fc_max],
                      "AM1-T1A-C2A": [angle_tail_max, angle_fc_max],
                      #"T1A-C2A-C3A": [angle_tail_max, angle_fc_max],
                      "AM2-C1B-C2B": [angle_tail_max, angle_fc_max]
                     }

X_cont_lowerbounds = {"PO4-AM1": [0.25, bond_fc_min], 
                      "AM1-AM2": [0.20, bond_fc_min], 
                      "AM1-T1A": [0.40, bond_fc_min], 
                      "AM2-C1B": [0.25, bond_fc_min],
                      "PO4-AM1-AM2": [angle_min, angle_fc_min],                              
                      "PO4-AM1-T1A": [angle_min, angle_fc_min],
                      "AM1-T1A-C2A": [angle_tail_min, angle_fc_min],
                      #"T1A-C2A-C3A": [angle_tail_min, angle_fc_min],
                      "AM2-C1B-C2B": [angle_tail_min, angle_fc_min]
                     }


filename_MDP_EM          = 'minim_new-rf.mdp'
filename_MDP_NPT         = 'npt_production.mdp'
filename_MDP_NPT_equil   = 'npt_equil.mdp'
filename_MDP_NPT_smalldt = 'npt_smalldt.mdp'

filename_MDP_NPT_Tm         = 'npt_Tm_biphasic_production.mdp'
filename_MDP_NPT_equil_Tm   = 'npt_Tm_biphasic_equil.mdp'
filename_MDP_NPT_smalldt_Tm = 'npt_Tm_biphasic_smalldt.mdp'

filename_MDP_NPT_Tm_base         = 'npt_Tm_biphasic_production_base.mdp'
filename_MDP_NPT_equil_Tm_base   = 'npt_Tm_biphasic_equil_base.mdp'
filename_MDP_NPT_smalldt_Tm_base = 'npt_Tm_biphasic_smalldt_base.mdp'

# Output #
name_output_production              = "production"
name_output_production_Tm           = "production_Tm"
filename_observables_dict           = "observables_dict.dat"

# Simulation #
timeout_Production_MDRUN            = 60*60*4 # In seconds

# energy minimization
nsteps_EM               = 10000
# short timestep equilibration
t_equil_smalldt         = 50
dt_small                = 0.002 # this has to be set in mdp file, here just for calc of nsteps
nsteps_smalldt          = int(t_equil_smalldt / dt_small)

t_equil                 = 50 # picoseconds
t_prod                  = 3000 # 75000
dt                      = 0.02
nsteps_equil            = int(t_equil / dt)


nsteps                  = int(t_prod / dt)
#print(nsteps)

t_equil_analysis = 0
t_start_rdf = 0 # 25000
dt_traj = 100
rdf_start_frame = int(t_start_rdf / dt_traj)

# Set what kind of data is kept.
WRITE_GRO       = False
WRITE_NDX       = False
WRITE_TPR       = False
WRITE_LOG       = False
WRITE_EDR       = True
WRITE_XTC       = False
WRITE_TRR       = False
SAVE_EQUIL      = False

bSaveVelocity   = True
bSaveParams     = True

T_list = np.array([286, 291, 296, 301, 311, 316, 321]) 
T_fluid_init     = 311
T_gel_init       = 260
T_penalty        = 10

t_melt_equil     = 100
t_melt_prod      = 3000 
nsteps_melt_equil = int(t_melt_equil / dt)
nsteps_melt_prod = int(t_melt_prod / dt)
nsim_melt = 1
t_start_melt    = 1000
t_stop_melt     = 3000
bUseWeightAPLFit = False


nsteps_dict = {
        'DPSM128_328K': nsteps,
        'POPC_SSM_CHOL': nsteps,
}

reverse_sorting = True # should be False if fitness is maximized, True if minimized             

# PSO parameters
w_init          = 0.5
alpha_init      = 0.5
bAdaptiveParameters     = True
bAdaptiveW              = True
bAdaptiveAlpha          = False
#w       = 0.25
c       = 2.0 #0.5
c_global = 2.0
#boundary_break = 0.05
v_max_scale = 0.2
gamma = 0.25

v_max = {}
for key in X_cont_upperbounds:
        v_max[key] = np.zeros(2)
        for j in range(2):
                v_max[key][j] = v_max_scale * (X_cont_upperbounds[key][j] - X_cont_lowerbounds[key][j])

print(v_max)

## Analysis
n_bins_bond_histo = 120
bond_histo_min = 0.1 # nm
bond_histo_max = 0.7 # nm

n_bins_angle_histo = 100
angle_histo_min = 0   # deg
angle_histo_max = 180 # deg

lipidnames      = {"POPC_SSM_CHOL":     ["POPC", "DPSM", "CHOL"],
                   "DPSM128_328K":      ['DPSM'],
                   "DPSM256_biphasic":  ['DPSM'],

                  }

selections_thickness = {
        "POPC_SSM_CHOL": {
                "leaflets": 'name AM1 AM2 GL1 GL2',
                "leaflet_filter": 'resname DPSM POPC',
                "thickness": 'resname DPSM POPC and name PO4'
        },
        "DPSM128_328K": {
                "leaflets": 'name AM1 AM2',
                "leaflet_filter": 'resname DPSM',
                "thickness": 'resname DPSM and name PO4'
        },
}


average_area_per_lipid_target = {
        'DPSM': {
                "POPC_SSM_CHOL":       0.485,   
                "DPSM128":            0.60, # PSM T = 45C Doktorova et al. J Phys Chem B 2020, 10.1021/acs.jpcb.0c03389
                "DPSM128_328K":       0.619 # PSM T = 55C Doktorova et al. J Phys Chem B 2020, 10.1021/acs.jpcb.0c03389
        },
}

average_area_per_lipid_tol = 0.015
average_area_per_lipid_cap = 0.3

# These values have to be checked, which bead selection matches literature, also still unclear.
thickness_target = {
        'DPSM': {
                "POPC_SSM_CHOL": 4.28,   
                "DPSM128": 3.84,         # D_B PSM T = 45C Doktorova et al.  J Phys Chem B 2020, 10.1021/acs.jpcb.0c03389
                "DPSM128_328K": 3.75    # D_B PSM T = 55C Doktorova et al. J Phys Chem B 2020, 10.1021/acs.jpcb.0c03389
        },
}


thickness_tol = 0.015 # corresponds to 1.5% tolerance
thickness_cap = 0.3

Tm_target = {
        'DPSM': {
                'DPSM256_biphasic': 313.85 # = 40.7C from Arsov et al. 10.1016/j.chemphyslip.2018.03.003
        }
}
Tm_tol = 0.015931177314003505 # corresponds to 5 K deviation around 313.85 K
Tm_cap = 0.5



#bonds_dist_target
target_bonds = {
        'DPSM': [
                "NC3-PO4", 
                "PO4-AM1", 
                "AM1-AM2",
                "AM1-T1A", 
                "T1A-C2A", 
                "C2A-C3A", 
                #"C3A-C4A", # extra bead in AA mapping
                "AM2-C1B", 
                "C1B-C2B", 
                "C2B-C3B", 
                "C3B-C4B"
        ],
        'DPCE': [
                "AM1-AM2",
                "AM1-T1A", 
                "T1A-C2A", 
                "C2A-C3A", 
                #"C3A-C4A", # extra bead in AA mapping
                "AM2-C1B", 
                "C1B-C2B", 
                "C2B-C3B", 
                "C3B-C4B"
        ]
}

mapping = '3bd'



# bond_data_popc_ssm_chol = read_distribution_file(
#         System.get_file_dir(__file__) + '/data/ref_data/POPC_SSM_CHOL/bond_lengths_%s.txt' %mapping,
#         target_bonds['DPSM'],
#         bins=n_bins_bond_histo,
#         range=(bond_histo_min, bond_histo_max)
#         )

# bond_data_ssm128_328K = read_distribution_file(
#         System.get_file_dir(__file__) + '/data/ref_data/SSM128_328K/bond_lengths_%s.txt' %mapping,
#         target_bonds['DPSM'],
#         bins=n_bins_bond_histo,
#         range=(bond_histo_min, bond_histo_max)
#         )

bond_data_popc_ssm_chol = json.load(
    open(System.get_file_dir(__file__) + '/data/ref_data/POPC_SSM_CHOL/bonddata.txt', 'r')
)

bond_data_ssm128_328K = json.load(
    open(System.get_file_dir(__file__) + '/data/ref_data/SSM128_328K/bonddata.txt', 'r')
)

bonds_dist_target = {
        'DPSM':{
                'POPC_SSM_CHOL': bond_data_popc_ssm_chol[1],
                'DPSM128_328K': bond_data_ssm128_328K[1],
        },
}

target_angles = {
        'DPSM': [
                "PO4-AM1-AM2", # 2 3 4
                "PO4-AM1-T1A", # 2 3 5
                "AM1-T1A-C2A", # 3 5 6
                "T1A-C2A-C3A", # 5 6 7
                "AM2-C1B-C2B", # 4 8 9
                "C1B-C2B-C3B", # 8 9 10
                "C2B-C3B-C4B"  # 9 10 11    
        ],
        'DPCE': [
                "AM1-T1A-C2A", # 1 3 4
                "T1A-C2A-C3A", # 3 4 5
                "AM2-C1B-C2B", # 2 6 7
                "C1B-C2B-C3B", # 6 7 8
                "C2B-C3B-C4B"  # 7 8 9
        ],
}


# angle_data_popc_ssm_chol = read_distribution_file(
#         System.get_file_dir(__file__) + '/data/ref_data/POPC_SSM_CHOL/angles_%s.txt' %mapping,
#         target_angles['DPSM'],
#         bins=n_bins_angle_histo,
#         range=(angle_histo_min, angle_histo_max)
# )


# angle_data_ssm128_328K = read_distribution_file(
#         System.get_file_dir(__file__) + '/data/ref_data/SSM128_328K/angles_%s.txt' %mapping,
#         target_angles['DPSM'],
#         bins=n_bins_angle_histo,
#         range=(angle_histo_min, angle_histo_max)
# )

angle_data_popc_ssm_chol = json.load(
    open(System.get_file_dir(__file__) + '/data/ref_data/POPC_SSM_CHOL/angledata.txt', 'r')
)

angle_data_ssm128_328K = json.load(
    open(System.get_file_dir(__file__) + '/data/ref_data/SSM128_328K/angledata.txt', 'r')
)

angles_dist_target = {
        'DPSM': {
                
                'POPC_SSM_CHOL': angle_data_popc_ssm_chol[1],
                'DPSM128_328K': angle_data_ssm128_328K[1],
        },
}

rdf_popc_ssm_chol_target = np.loadtxt(System.get_file_dir(__file__) + '/data/ref_data/POPC_SSM_CHOL/rdf_SSM_CHOL_2d_1p5nm_30bins.dat')
rdf_popc_ssm_chol_target_bins = np.ascontiguousarray(rdf_popc_ssm_chol_target[:,0])
rdf_popc_ssm_chol_target_rdf = np.ascontiguousarray(rdf_popc_ssm_chol_target[:,1]) # slices are not C-contiguous, pyemd needs that

rdf_popc_ssm_chol_target_POPC = np.loadtxt(System.get_file_dir(__file__) + '/data/ref_data/POPC_SSM_CHOL/rdf_SSM_POPC_2d_1p5nm_30bins.dat')
rdf_popc_ssm_chol_target_POPC_bins = np.ascontiguousarray(rdf_popc_ssm_chol_target_POPC[:,0])
rdf_popc_ssm_chol_target_POPC_rdf = np.ascontiguousarray(rdf_popc_ssm_chol_target_POPC[:,1]) # slices are not C-contiguous, pyemd needs that

rdf_2d_target = {
        'DPSM': {
                'POPC_SSM_CHOL': {
                        'DPSM_CHOL': [ rdf_popc_ssm_chol_target_rdf, rdf_popc_ssm_chol_target_bins ],
                        'DPSM_POPC': [ rdf_popc_ssm_chol_target_POPC_rdf, rdf_popc_ssm_chol_target_POPC_bins ]
                        }, # emd_score expects [hist, bins]
                'DPSM128': {
                        'DPSM_CHOL': [np.zeros(10), np.linspace(0, 1, num=10)],
                        'DPSM_POPC': [np.zeros(10), np.linspace(0, 1, num=10)]
                        } 
        }
} 

rdf_beads2D_target = {
        "DPSM": {
                'DPSM128:': {
                        'PO4-PO4': [],
                        'AM1-AM1': [],
                        'AM1-AM2': [],
                        'AM2-AM2': [],
                },
        },
}


# observable target dicts are structured {molecule: {training_system: data }}
target_dict = {
        'DPSM': {
                'average_area_per_lipid': average_area_per_lipid_target,
                'thickness': thickness_target,
                'Tm': Tm_target,
                'bond_lengths_dist': bonds_dist_target,
                'angles_dist': angles_dist_target,
                'rdf_2d': rdf_2d_target,
        },
}   

## overall weight of each observable
scoring_weights_observables = {
        'bond_lengths_dist':            1,
        'angles_dist':                  100,
        'rdf_2d':                       1,
        'thickness':                   500,
        'average_area_per_lipid':      1000,
        'Tm':                          250,
}

# Relative weight per observable of each system. Weights are summed and cost is normalized by sum.
# Systems that are defined in the weights dict but not used, i.e., not in training_systems are
# ignored in the normalization.
# To balance importance due to quality in experimental data or sampling.  
# This is also used to determine which observable is used in the cost calculation.
scoring_weights_tr_systems = {
        'DPSM': {
                'DPSM128_328K': {
                        'average_area_per_lipid':       1,
                        'thickness':                    1, 
                        'bond_lengths_dist':            1,
                        'angles_dist':                  1,
                        #'rdf_2d':                       1
                },
                'POPC_SSM_CHOL':{
                        'average_area_per_lipid':       1, 
                        'thickness':                    1,  
                        'bond_lengths_dist':            1,
                        'angles_dist':                  1,
                        'rdf_2d':                       1
                },
                'DPSM256_biphasic': {
                        'Tm':                            1
                },
        }
}

range_SAE_apl = partial(
        ScoringFunctions.range_SAE, 
        Etol=average_area_per_lipid_tol, 
        cap=average_area_per_lipid_cap)


range_SAE_thickness = partial(
        ScoringFunctions.range_SAE, 
        Etol=thickness_tol, 
        cap=thickness_cap)

range_SAE_Tm = partial(
        ScoringFunctions.range_SAE, 
        Etol=Tm_tol, 
        cap=Tm_cap)


scoring_functions_dict = {
        'average_area_per_lipid':       range_SAE_apl, 
        'thickness':                    range_SAE_thickness, 
        'Tm':                           range_SAE_Tm, 
        #'bond_lengths_av':              ScoringFunctions.bonds_SAE,
        #'angles_av':                    ScoringFunctions.angles_SAE,
        'bond_lengths_dist':            ScoringFunctions.emd_score,
        'angles_dist':                  ScoringFunctions.emd_score,
        'rdf_2d':                       ScoringFunctions.emd_score,

}      

bCalcObservable = {
        'DPSM128_328K': {
                'average_area_per_lipid': True,
                'thickness': True,
                'bond_lengths_av': True,
                'angles_av': True,
                'bond_lengths_dist': True,
                'angles_dist': True,
                'rdf_2d': False,
        },
        'POPC_SSM_CHOL': {
                'average_area_per_lipid': True,
                'thickness': True,
                'bond_lengths_av': True,
                'angles_av': True,
                'bond_lengths_dist': True,
                'angles_dist': True,
                'rdf_2d': True,
        },
        'DPSM256_biphasic': {
                'average_area_per_lipid': False,
                'thickness': False,
                'bond_lengths_av': True,
                'angles_av': True,
                'bond_lengths_dist': True,
                'angles_dist': True,
                'Tm': True,
                'rdf_2d': False,
        },

}

check_observables_calc(scoring_weights_tr_systems, bCalcObservable)



mda_backend  = 'serial' # ['serial', 'OpenMP']
useGPU = True #False #True
updateGPU = False #True # does not work with virtual sites

resnm_to_indexgroup_dict            = {
                                        "POPC"  : ("System", "POPC", "Lipids"),
                                        "DOPC"  : ("System", "DOPC", "Lipids"),
                                        "POPE"  : ("System", "POPE", "Lipids"),
                                        "POPS"  : ("System", "POPS", "Lipids"),
                                        "DPSM"  : ("System", "DPSM", "Lipids"),
                                        "DPCE"  : ("System", "DPCE", "Lipids"),
                                        "CHOL"  : ("System", "CHOL", "Lipids"),
                                        "W"     : ("System", "Solvent"), 
                                        "NA+"   : ("System", "Solvent"), 
                                        "CL-"   : ("System", "Solvent"),
                                        "ION"   : ("System", "Solvent"),
                                        "NA"    : ("System", "Solvent"), 
                                        "CL"    : ("System", "Solvent"),
                                        # "GLY"   : "Protein", "ALA"  : "Protein", "ASP"  : "Protein", "ASN"  : "Protein", "GLU"  : "Protein",
                                        # "GLN"   : "Protein", "VAL"  : "Protein", "LEU"  : "Protein", "ILE"  : "Protein", "MET"  : "Protein",
                                        # "THR"   : "Protein", "SER"  : "Protein", "CYS"  : "Protein", "LYS"  : "Protein", "ARG"  : "Protein",
                                        # "HIS"   : "Protein", "PHE"  : "Protein", "PRO"  : "Protein", "TRP"  : "Protein", "TYR"  : "Protein"      
                                      }