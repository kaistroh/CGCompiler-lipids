from curses import init_pair
from statistics import mean
import pandas as pd
import numpy as np
import MDAnalysis as mda
import panedr
from lmfit.models import LinearModel
from lipyphilic.lib.assign_leaflets import AssignLeaflets
from lipyphilic.lib.memb_thickness import MembThickness
from lipyphilic.lib.neighbours import Neighbours

from user.analysis.utils import get_number_of_lipids, APLTemperatureModel, AICc
from user.analysis.utils import edr2df_list, line, makeAtomGroupList, min_image_dist
import user.usersettings as UserSettings


def average_area_per_lipid(u, edr, lipidnames):
    n_lipids_per_leaflet = get_number_of_lipids(u, lipidnames) / 2
    edr_df = panedr.edr_to_df(edr)

    t_equil = UserSettings.t_equil_analysis

    box_x_mean = np.mean(edr_df[edr_df['Time'] >= t_equil]['Box-X'].values)
    box_y_mean = np.mean(edr_df[edr_df['Time'] >= t_equil]['Box-Y'].values)
    
    return box_x_mean * box_y_mean / n_lipids_per_leaflet


def bond_lengths(u, molecule_name, bonds_to_optimize):
    n_molecules = u.select_atoms('resname %s' %molecule_name).n_residues
    n_bonds = len(bonds_to_optimize)

    atomgroups = {}
    for bond in bonds_to_optimize:
        a0_name, a1_name = bond.split('-')
        selection0 = 'resname %s and name %s' %(molecule_name, a0_name)
        ag0 = u.select_atoms(selection0)
        selection1 = 'resname %s and name %s' %(molecule_name, a1_name)
        ag1 = u.select_atoms(selection1)
        atomgroups[bond] = [ag0, ag1]

    bond_lengths = np.zeros((len(u.trajectory) * n_molecules, n_bonds))
    bond_lengths_frame = np.zeros(n_molecules)

    for t_ndx, ts in enumerate(u.trajectory):
        for i, bond in enumerate(bonds_to_optimize):
            mda.lib.distances.calc_bonds(
                atomgroups[bond][0].positions,
                atomgroups[bond][1].positions,
                box=u.dimensions,
                result=bond_lengths_frame,
                backend=UserSettings.mda_backend
            )
        
            bond_lengths[t_ndx*n_molecules:(t_ndx+1)*n_molecules, i] = bond_lengths_frame / 10

    average_bond_length = {}
    bond_length_distribution = {}
    for i, bond in enumerate(bonds_to_optimize):
        average_bond_length[bond] = np.mean(bond_lengths[:,i])
        bond_length_distribution[bond] = np.histogram(
            bond_lengths[:,i],
            bins=UserSettings.n_bins_bond_histo,
            range=(UserSettings.bond_histo_min, UserSettings.bond_histo_max), 
            density=True
        )

    return average_bond_length, bond_length_distribution

def angles(u, molecule_name, angles_to_optimize):
    n_molecules = u.select_atoms('resname %s' %molecule_name).n_residues
    n_angles = len(angles_to_optimize)

    atomgroups = {}
    for angle in angles_to_optimize:
        a0_name, a1_name, a2_name = angle.split('-')
        
        selection0 = 'resname %s and name %s' %(molecule_name, a0_name)
        selection1 = 'resname %s and name %s' %(molecule_name, a1_name)
        selection2 = 'resname %s and name %s' %(molecule_name, a2_name)
        ag0 = u.select_atoms(selection0)
        ag1 = u.select_atoms(selection1)
        ag2 = u.select_atoms(selection2)
        
        atomgroups[angle] = [ag0, ag1, ag2]

    angles = np.zeros((len(u.trajectory) * n_molecules, n_angles))
    angles_frame = np.zeros(n_molecules)

    for t_ndx, ts in enumerate(u.trajectory):
        for i, angle in enumerate(angles_to_optimize):
            mda.lib.distances.calc_angles(
                atomgroups[angle][0].positions,
                atomgroups[angle][1].positions,
                atomgroups[angle][2].positions,
                box=u.dimensions,
                result=angles_frame,
                backend=UserSettings.mda_backend
            )
        
            angles[t_ndx*n_molecules:(t_ndx+1)*n_molecules, i] = angles_frame

    average_angles = {}
    angle_distribution = {}
    for i, angle in enumerate(angles_to_optimize):
        average_angles[angle] = np.rad2deg(np.mean(angles[:,i]))
        angle_distribution[angle] = np.histogram(
            np.rad2deg(angles[:,i]), 
            bins=UserSettings.n_bins_angle_histo, 
            range=(UserSettings.angle_histo_min, UserSettings.angle_histo_max), 
            density=True
        )

    return average_angles, angle_distribution

def membrane_thickness(u, selections):
    leaflets = AssignLeaflets(
        universe=u,
        lipid_sel=selections['leaflets']
    )
    leaflets.run()

    memb_thickness = MembThickness(
        universe=u,
        leaflets=leaflets.filter_leaflets(selections['leaflet_filter']),
        lipid_sel=selections['thickness'],
        return_surface=False
    )

    memb_thickness.run(
        start=None,
        stop=None,
        step=None,
        verbose=False
    )

    return np.mean(memb_thickness.memb_thickness) / 10


def melting_temperature_biphasic_apl_wModelSelection(EDR_file_dict, 
    T_list, nsim, u, lipidnames, t_start=20, t_stop=2000, 
    bUseWeights=False,
    T_penalty=0):
    n_lipids_per_leaflet = get_number_of_lipids(u, lipidnames) / 2
    edr_dict = {}
    grp_dict = {}
    for T in T_list:
        edrs = pd.concat(edr2df_list(EDR_file_dict[T]), keys=range(nsim))
        edrs.index.set_names(['sim_ndx', 'time'], inplace=True)
        edrs['APL'] = edrs['Box-X'] * edrs['Box-Y'] / n_lipids_per_leaflet
        edr_dict[T] = edrs
        grp_dict[T] = edrs.groupby('sim_ndx')

    mean_apls = np.zeros((len(T_list), nsim))
    init_apls = np.zeros((len(T_list), nsim))

    for T_ndx, T in enumerate(T_list):
        for sim_ndx, frame in grp_dict[T]:
            mask = (frame.loc[:,'Time'] >= t_start) & (frame.loc[:,'Time'] <= t_stop)
            mask_t0 = (frame.loc[:,'Time'] == 0) 
            apl = frame[mask]['APL']
            mean_apls[T_ndx, sim_ndx] = apl.mean()
            init_apls[T_ndx, sim_ndx] = frame[mask_t0]['APL']

    apls_average = mean_apls.mean(axis=1)
    if nsim > 1:
        apls_err = mean_apls.std(axis=1, ddof=1) / np.sqrt(nsim)
    else:
        apls_err = np.full_like(apls_average, np.nan)
    apl_threshold = np.mean(init_apls)
    if nsim != 1 and bUseWeights:   
        weights = 1 / apls_err
    else:
        weights=None

    model = APLTemperatureModel()
    params = model.guess(T_list, apls_average)
    fit = model.fit(
        apls_average, 
        params, 
        T=T_list, 
        weights=weights
    )

    lin_model = LinearModel()
    lin_fit = lin_model.fit(
        apls_average,
        x=T_list,
        weights=weights
    )

    criterion_sig = AICc(fit.aic, fit.ndata, fit.nvarys)
    criterion_lin = AICc(lin_fit.aic, lin_fit.ndata, lin_fit.nvarys)

    if criterion_sig < criterion_lin:    
        better_model = 'SIGMOIDAL'
        Tm = fit.best_values['Tm']
    else:
        better_model = "LINEAR"
        if np.min(apls_average) >= apl_threshold:
            Tm = np.min(T_list) - T_penalty
        elif np.max(apls_average) <= apl_threshold:
            Tm = np.max(T_list) + T_penalty
        else:
            Tm = fit.best_values['Tm'] ## 

    return (Tm.astype('float'), fit, fit.best_values, apls_average, apls_err, apl_threshold, better_model,
            criterion_sig, criterion_lin)


def get_apl_mean(grp, n_lipids_per_leaflet, t_start, t_stop, stride=1):
    mask = (grp.loc[:,'Time'] >= t_start) & (grp.loc[:,'Time'] <= t_stop)
    apl = grp[mask]['Box-X'] * grp[mask]['Box-Y'] / n_lipids_per_leaflet
    
    return apl.mean()


def enrichment_index_groups(u, selection, cutoff, start_frame=None, stop_frame=None, step=None):
    neighbors = Neighbours(
        universe=u,
        lipid_sel=selection,
        cutoff=cutoff
    )
    neighbors.run(start=start_frame, stop=stop_frame, step=step, verbose=False)

    counts, enrichment = neighbors.count_neighbours(return_enrichment=True)
    enrichment['Time'] = enrichment['Frame'] * u.trajectory.dt
    enrichment_grps = enrichment.groupby('Label')

    return enrichment_grps

def enrichment_index(enrichment_grps, lipid1, lipid2, t_analysis_start=0):
    
    grp = enrichment_grps.get_group(lipid1)
    mask = grp['Time'] >= t_analysis_start

    mean_enrichment = grp[mask][lipid2].mean()

    return mean_enrichment

def calcCosThetaSquare(ag0, ag1, normal, box):
    coords0 = ag0.positions / 10
    coords1 = ag1.positions / 10
    
    b = min_image_dist(coords1, coords0, box)

    return np.dot(b, normal)**2 / np.einsum('ij,ij->i', b, b)

def calcSzz(ag0, ag1, normal, box):
    return 0.5 * np.mean(3 * calcCosThetaSquare(ag0, ag1, normal, box) - 1)

def calcTailOrder(
        u, 
        selection, 
        tails, 
        normal_vec=np.array([0,0,1]), 
        start=None, 
        stop=None, 
        step=None, 
        updating_selection=True
    ):
    lipids = u.select_atoms(selection)

    groups = {}
    for tail_key in tails:
        groups[tail_key] = makeAtomGroupList(lipids, tails[tail_key], updating_selection)

    columns = {}
    for tail_key in tails:
        columns[tail_key] = []
        for i in range(len(tails[tail_key]) - 1):
            columns[tail_key].append('%s-%s' %(tails[tail_key][i], tails[tail_key][i+1]))
    
    n_cols = {}
    nDataFields = 1
    for tail_key in columns:
        n_cols[tail_key] = len(columns[tail_key])
        nDataFields += len(columns[tail_key])
    
    nFrames = (len(u.trajectory) - start) // step + 1
    print(nFrames)

    columns_list = ['time']
    for tail_key in columns:
        columns_list += columns[tail_key]
    print(columns_list)
    Szz = pd.DataFrame(np.zeros((nFrames, nDataFields)),
                        columns=columns_list)

    for i, frame in enumerate(u.trajectory[start::step]):
        Szz['time'][i] = frame.time
        box = u.dimensions[:3] / 10
        for tail_key in tails:
            for j, column in enumerate(columns[tail_key]):
                Szz.iloc[i][column] = calcSzz(groups[tail_key][j],
                                                groups[tail_key][j+1],
                                                normal_vec, 
                                                box)

    Szz.set_index('time', inplace=True)
    
    return Szz