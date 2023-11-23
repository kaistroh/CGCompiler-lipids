import pandas as pd
import numpy as np
import numpy.linalg as npla

import MDAnalysis as mda
import panedr
import lmfit


def min_image_dist(a, b, box):
    d = a - b
    return d - box * np.rint(d / box)


def get_number_of_lipids(u, lipidnames):
    lipids = u.select_atoms('resname %s' %(' '.join(lipidnames)))
    return lipids.n_residues

def read_distribution_file(filename, keys, bins, range):
    data = np.loadtxt(filename)

    averages = {}
    distributions = {}
    for i, key in enumerate(keys):
        averages[key] = np.mean(data[:,i])
        distributions[key] = np.histogram(
            data[:,i],
            bins=bins,
            range=range,
            density=True
        )

    return (averages, distributions)

def center_of_mass_special_treatment(ag):
    coms = np.zeros((len(ag.residues), 3))
    
    for i, res in enumerate(ag.residues):
        coords = res.atoms.unwrap(compound='fragments', reference='cog', inplace=True)
        coms[i,:] = res.atoms.center_of_mass()
        
    return coms

def project_coords_array_on_plane(coords, n):
    return coords - (np.dot(coords, n) / npla.norm(n)**2).reshape(-1,1) * n

def edr2df_list(filenames):
    df_list = []
    for filename in filenames:
        df_list.append(panedr.edr_to_df(filename))

    return df_list

def line(x, a, m):
    return a + x * m 

def get_times(u, start=None, stop=None, step=None):
    times = []
    for ts in u.trajectory[start:stop:step]:
        times.append(ts.time)
        
    return np.array(times)


def local_resindices(ag):
    loc_indices = np.zeros(ag.n_atoms, dtype=np.int64)
    
    r_ndx = 0
    a_ndx = 0
    for res in ag.residues:
        for atom in res.atoms:
            if atom in ag:
                loc_indices[a_ndx] = r_ndx
                a_ndx += 1

        r_ndx += 1
        
    return loc_indices


def makeAtomGroupList(parentgroup, atomnames, updating=True):
    ag_list = []
    for name in atomnames:
        ag = parentgroup.select_atoms('name %s' %name, updating=False)
        ag_list.append(ag)
        
    return ag_list


def ndx_to_dict(ndxfile):
    #ag_dict = {}
    group_name = None
    indices = []
    for line in ndxfile:
        comment_idx = line.find(';')
        if comment_idx >= 0:
            line = line[comment_idx:]
        line = line.strip()
        if line.startswith('['):
            if group_name is not None:
                indices = np.array(indices, dtype=int) - 1
                yield (group_name, indices)
            group_name = line[1:-1].strip()
            indices = []
        else:
            indices += line.split()
    if group_name is not None:
        indices = np.array(indices, dtype=int) - 1
        yield (group_name, indices)

def make_ndx_dict(ndxfile):
    ndx_dict = {}
    with open(ndxfile) as infile:
        for group_name, indices in ndx_to_dict(infile):
            print(group_name, indices)
            ndx_dict[group_name] = indices
    
    return ndx_dict

def make_ag_dict(ag, ndx_dict):
    ag_dict = {}
    for bead in ndx_dict:
        ag_dict[bead] = []
        for lipid in ag.fragments:
            print(bead, lipid.atoms[ndx_dict[bead]])
            #lipid.atoms[nd]
            ag_dict[bead].append(lipid.atoms[ndx_dict[bead]])
            
    return ag_dict


def unionize_ag(ag_list):
    ag_union = ag_list[0]
    if len(ag_list) > 1:
        for ag in ag_list[1:]:
            ag_union = ag_union.union(ag)
            
    return ag_union


class APLTemperatureModel(lmfit.Model):
    def __init__(self, *args, **kwargs):
        def enthalpy_temperature(T, APL_0, c_p_g, dAPL, dc_p, k, Tm):
            return APL_0 + c_p_g * T + (dAPL + dc_p * (T - Tm)) * 1 / (1 + np.exp(-k * (T - Tm)))
        super().__init__(enthalpy_temperature, *args, **kwargs)
    
    def guess(self, tdata, hdata, **kwargs):
        params = self.make_params()
        def pset(param, value, min=None, max=None, vary=True):
            params["%s%s" %(self.prefix, param)].set(value=value, min=min, max=max, vary=vary)
            
        pset("APL_0", np.min(hdata) - 0.0001 * 276, 0., 0.6)
        pset("c_p_g", 1e-4, 0, 1e-2)
        pset("dAPL", np.max(hdata) - np.min(hdata), 0.05, 0.15)
        pset("dc_p", 0, vary=False)
        pset('k', 1, 0.1, 2)
        pset('Tm', tdata[int(len(tdata) / 2)], 271, 326)
        #pset('T_m', tdata[find_step_Window(hdata)[0]], tdata.min(), tdata.max())         
        
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

def AICc(aic, n, k):
    return aic + (2 * k**2 + 2*k) / (n - k - 1)