import numpy as np
from copy import deepcopy

from user.analysis import utils

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis import distances

from lipyphilic.lib.assign_leaflets import AssignLeaflets

from .utils import project_coords_array_on_plane, local_resindices

class RDF2dCOMperLeaflet(AnalysisBase):


    def __init__(self, g1, g2,
                 nbins=75, range=(0.0, 15.0), exclusion_block=None, virtual_sites=(False, False),
                 leaflets_lipid_selection='name GL1 GL2 AM1 AM2 ROH',
                 n_bins_leaflet=1,
                 g1_selection=None,
                 g2_selection=None,
                 start_frame=None,
                 end_frame=None,
                 frame_stride=None,
                 normal_vector=np.array([0,0,1]),
                 **kwargs):
        super(RDF2dCOMperLeaflet, self).__init__(g1.universe.trajectory, **kwargs)
        self.g1 = g1
        self.g2 = g2
        self.u = g1.universe

        self.rdf_settings = {'bins': nbins,
                             'range': range}
        self._exclusion_block = exclusion_block
        self._virtual_sites = virtual_sites

        self.leaflets_lipid_selection = leaflets_lipid_selection
        self.n_bins_leaflet = n_bins_leaflet 
        self.g1_selection = g1_selection
        self.g2_selection = g2_selection
        self.leaflets_start = start_frame
        self.leaflets_end = end_frame
        self.leaflets_step = frame_stride

        self.normal_vector = normal_vector

    def _prepare(self):
        # Empty histogram to store the RDF
        count, edges = np.histogram([-1], **self.rdf_settings)
        count = count.astype(np.float64)
        count *= 0.0
        self.results.count_up = deepcopy(count)
        self.results.count_lo = deepcopy(count)
        self.results.edges = edges
        self.results.bins = 0.5 * (edges[:-1] + edges[1:])

        # Need to know average area
        self.area = 0.0
        self.density_up = 0.0
        self.density_lo = 0.0
        self._maxrange = self.rdf_settings['range'][1]

        self.leaflets = AssignLeaflets(
            universe=self.u,
            lipid_sel=self.leaflets_lipid_selection,
            n_bins=self.n_bins_leaflet
        )
        self.leaflets.run(self.leaflets_start, self.leaflets_end, self.leaflets_step)

        self.g1_leaflets = self.leaflets.filter_leaflets(self.g1_selection)
        self.g2_leaflets = self.leaflets.filter_leaflets(self.g2_selection)

    def _single_frame(self):
        if self._virtual_sites[0]:
            g1_coms = utils.center_of_mass_special_treatment(self.g1)
        else:
            g1_coms = self.g1.center_of_mass(unwrap=True, compound='molecules')

        if self._virtual_sites[1]:
            g2_coms = utils.center_of_mass_special_treatment(self.g2)
        else:
            g2_coms = self.g2.center_of_mass(unwrap=True, compound='molecules')

        g1_up_mask = self.g1_leaflets[:,self._frame_index] == 1
        g2_up_mask = self.g2_leaflets[:,self._frame_index] == 1

        g1_lo_mask = self.g1_leaflets[:,self._frame_index] == -1
        g2_lo_mask = self.g2_leaflets[:,self._frame_index] == -1

        pairs_up, dist_up = distances.capped_distance(
            project_coords_array_on_plane(g1_coms[g1_up_mask], self.normal_vector),
            project_coords_array_on_plane(g2_coms[g2_up_mask], self.normal_vector),
            self._maxrange,
            box=self.u.dimensions
        )

        pairs_lo, dist_lo = distances.capped_distance(
            project_coords_array_on_plane(g1_coms[g1_lo_mask], self.normal_vector),
            project_coords_array_on_plane(g2_coms[g2_lo_mask], self.normal_vector),
            self._maxrange,
            box=self.u.dimensions
        )

        # Exclusions
        if self._exclusion_block is not None:
            idxA_up = pairs_up[:,0]//self._exclusion_block[0]
            idxB_up = pairs_up[:,1]//self._exclusion_block[1]
            mask_up = np.where(idxA_up != idxB_up)[0]
            dist_up = dist_up[mask_up]

            idxA_lo = pairs_lo[:,0]//self._exclusion_block[0]
            idxB_lo = pairs_lo[:,1]//self._exclusion_block[1]
            mask_lo = np.where(idxA_lo != idxB_lo)[0]
            dist_lo = dist_lo[mask_lo]

        count_up = np.histogram(dist_up, **self.rdf_settings)[0]
        self.results.count_up += count_up

        count_lo = np.histogram(dist_lo, **self.rdf_settings)[0]
        self.results.count_lo += count_lo

        n_g1_up = np.sum(self.g1_leaflets[:,self._frame_index] == 1)
        n_g2_up = np.sum(self.g2_leaflets[:,self._frame_index] == 1)
        N_up = n_g1_up * n_g2_up

        n_g1_lo = np.sum(self.g1_leaflets[:,self._frame_index] == -1)
        n_g2_lo = np.sum(self.g2_leaflets[:,self._frame_index] == -1)
        N_lo = n_g1_lo * n_g2_lo
        
        # Exclusions 
        if self._exclusion_block:
            xA, xB = self._exclusion_block
            nblocks_up = n_g1_up / xA
            N_up -= xA * xB * nblocks_up

            nblocks_lo = n_g1_lo / xA
            N_lo -= xA * xB * nblocks_lo

        self.area += self.u.dimensions[0] * self.u.dimensions[1] # This would need to be changed for anything other than z-axis
        self.density_up += N_up / (self.u.dimensions[0] * self.u.dimensions[1])
        self.density_lo += N_lo / (self.u.dimensions[0] * self.u.dimensions[1])


    def _conclude(self):

        # Area in each shell
        areas = np.power(self.results.edges, 2)
        area = np.pi * np.diff(areas)

        av_density_up = self.density_up / self.n_frames
        av_density_lo = self.density_lo / self.n_frames

        print(av_density_up, av_density_lo)

        rdf_up = self.results.count_up / (av_density_up * area * self.n_frames)
        rdf_lo = self.results.count_lo / (av_density_lo * area * self.n_frames)

        self.results.rdf_up = rdf_up
        self.results.rdf_lo = rdf_lo

        self.results.rdf = 0.5 * (rdf_up + rdf_lo)

        self.results.n_of_r_up = np.pi * av_density_up * np.cumsum(self.results.rdf_up * area * np.diff(self.results.edges)[0])
        self.results.n_of_r_lo = np.pi * av_density_lo * np.trapz(self.results.rdf_lo * area, dx=np.diff(self.results.edges)[0])

        self.results.n_of_r = 0.5 * (self.results.n_of_r_up + self.results.n_of_r_lo)


class RDF2dperLeaflet(AnalysisBase):

    def __init__(self, g1, g2,
                 nbins=75, range=(0.0, 15.0), exclusion_block=None, virtual_sites=(False, False),
                 leaflets_lipid_selection='name GL1 GL2 AM1 AM2 ROH',
                 n_bins_leaflet=1,
                 g1_selection=None,
                 g2_selection=None,
                 start_frame=None,
                 end_frame=None,
                 frame_stride=None,
                 normal_vector=np.array([0,0,1]),
                 **kwargs):
        super(RDF2dperLeaflet, self).__init__(g1.universe.trajectory, **kwargs)
        self.g1 = g1
        self.g2 = g2
        self.u = g1.universe

        self.rdf_settings = {'bins': nbins,
                             'range': range}
        self._exclusion_block = exclusion_block
        self._virtual_sites = virtual_sites

        self.leaflets_lipid_selection = leaflets_lipid_selection
        self.n_bins_leaflet = n_bins_leaflet 
        self.g1_selection = g1_selection
        self.g2_selection = g2_selection
        self.leaflets_start = start_frame
        self.leaflets_end = end_frame
        self.leaflets_step = frame_stride

        self.normal_vector = normal_vector

    def _prepare(self):
        # Empty histogram to store the RDF
        count, edges = np.histogram([-1], **self.rdf_settings)
        count = count.astype(np.float64)
        count *= 0.0
        self.results.count_up = deepcopy(count)
        self.results.count_lo = deepcopy(count)
        self.results.edges = edges
        self.results.bins = 0.5 * (edges[:-1] + edges[1:])

        # Need to know average area
        self.area = 0.0
        self.density_up = 0.0
        self.density_lo = 0.0
        self._maxrange = self.rdf_settings['range'][1]

        self.leaflets = AssignLeaflets(
            universe=self.u,
            lipid_sel=self.leaflets_lipid_selection,
            n_bins=self.n_bins_leaflet
        )
        self.leaflets.run(self.leaflets_start, self.leaflets_end, self.leaflets_step)

        self.g1_leaflets = self.leaflets.filter_leaflets(self.g1_selection)
        self.g2_leaflets = self.leaflets.filter_leaflets(self.g2_selection)

    def _single_frame(self):
        if self._virtual_sites[0]:
            g1_coms = self.g1.positions 
        else:
            g1_coms = self.g1.positions 

        if self._virtual_sites[1]:
            g2_coms = self.g2.positions 
        else:
            g2_coms = self.g2.positions 

        g1_up_mask = self.g1_leaflets[:,self._frame_index] == 1
        g2_up_mask = self.g2_leaflets[:,self._frame_index] == 1

        g1_lo_mask = self.g1_leaflets[:,self._frame_index] == -1
        g2_lo_mask = self.g2_leaflets[:,self._frame_index] == -1

        pairs_up, dist_up = distances.capped_distance(
            project_coords_array_on_plane(g1_coms[g1_up_mask], self.normal_vector),
            project_coords_array_on_plane(g2_coms[g2_up_mask], self.normal_vector),
            self._maxrange,
            box=self.u.dimensions
        )

        pairs_lo, dist_lo = distances.capped_distance(
            project_coords_array_on_plane(g1_coms[g1_lo_mask], self.normal_vector),
            project_coords_array_on_plane(g2_coms[g2_lo_mask], self.normal_vector),
            self._maxrange,
            box=self.u.dimensions
        )
        #print(len(dist_up))
        #print(pairs_up)
        # Exclusions
        if self._exclusion_block is not None:
            idxA_up = pairs_up[:,0]//self._exclusion_block[0]
            idxB_up = pairs_up[:,1]//self._exclusion_block[1]
            mask_up = np.where(idxA_up != idxB_up)[0]
            dist_up = dist_up[mask_up]

            idxA_lo = pairs_lo[:,0]//self._exclusion_block[0]
            idxB_lo = pairs_lo[:,1]//self._exclusion_block[1]
            mask_lo = np.where(idxA_lo != idxB_lo)[0]
            dist_lo = dist_lo[mask_lo]
        #print(len(dist_up))
        #print(pairs_up)

        count_up = np.histogram(dist_up, **self.rdf_settings)[0]
        self.results.count_up += count_up

        count_lo = np.histogram(dist_lo, **self.rdf_settings)[0]
        self.results.count_lo += count_lo

        n_g1_up = np.sum(self.g1_leaflets[:,self._frame_index] == 1)
        n_g2_up = np.sum(self.g2_leaflets[:,self._frame_index] == 1)
        N_up = n_g1_up * n_g2_up
        #print(n_g1_up, n_g2_up)

        n_g1_lo = np.sum(self.g1_leaflets[:,self._frame_index] == -1)
        n_g2_lo = np.sum(self.g2_leaflets[:,self._frame_index] == -1)
        N_lo = n_g1_lo * n_g2_lo
        
        # # Exclusions 
        # if self._exclusion_block:
        #     xA, xB = self._exclusion_block
        #     nblocks_up = n_g1_up / xA
        #     N_up -= xA * xB * nblocks_up

        #     nblocks_lo = n_g1_lo / xA
        #     N_lo -= xA * xB * nblocks_lo

        #print(N_up, N_lo)
        self.area += self.u.dimensions[0] * self.u.dimensions[1] # This would need to be changed for anything other than z-axis
        self.density_up += N_up / (self.u.dimensions[0] * self.u.dimensions[1])
        self.density_lo += N_lo / (self.u.dimensions[0] * self.u.dimensions[1])

        #print(self.u.dimensions)

    def _conclude(self):

        # Area in each shell
        areas = np.power(self.results.edges, 2)
        area = np.pi * np.diff(areas)

        av_density_up = self.density_up / self.n_frames
        av_density_lo = self.density_lo / self.n_frames

        print(av_density_up, av_density_lo)

        rdf_up = self.results.count_up / (av_density_up * area * self.n_frames)
        rdf_lo = self.results.count_lo / (av_density_lo * area * self.n_frames)

        self.results.rdf_up = rdf_up
        self.results.rdf_lo = rdf_lo

        self.results.rdf = 0.5 * (rdf_up + rdf_lo)

class RDF2dAGperLeaflet(AnalysisBase):


    def __init__(self,
                 universe,
                 g1_selection, 
                 g2_selection,
                 nbins=75, 
                 range=(0.0, 15.0), 
                 exclusion_block=None, 
                 leaflets_lipid_selection='name GL1 GL2 AM1 AM2 ROH',
                 n_bins_leaflet=1,
                 g1_leaflet_filter=None,
                 g2_leaflet_filter=None,
                 start_frame=None,
                 end_frame=None,
                 frame_stride=None,
                 normal_vector=np.array([0,0,1]),
                 **kwargs):
        super(RDF2dAGperLeaflet, self).__init__(universe.trajectory, **kwargs)
        self.g1 = universe.select_atoms(g1_selection)
        self.g2 = universe.select_atoms(g2_selection)
        self.u = universe

        self.rdf_settings = {'bins': nbins,
                             'range': range}
        self._exclusion_block = exclusion_block
        #self._virtual_sites = virtual_sites

        self.leaflets_lipid_selection = leaflets_lipid_selection
        self.n_bins_leaflet = n_bins_leaflet 
        self.g1_selection = g1_selection
        self.g2_selection = g2_selection

        self.g1_leaflet_filter = g1_leaflet_filter
        self.g2_leaflet_filter = g2_leaflet_filter

        self.leaflets_start = start_frame
        self.leaflets_end = end_frame
        self.leaflets_step = frame_stride

        self.normal_vector = normal_vector

        self.g1_local_resindices = local_resindices(self.g1)
        self.g2_local_resindices = local_resindices(self.g2)

    def _prepare(self):
        # Empty histogram to store the RDF
        count, edges = np.histogram([-1], **self.rdf_settings)
        count = count.astype(np.float64)
        count *= 0.0
        self.results.count_up = deepcopy(count)
        self.results.count_lo = deepcopy(count)
        self.results.edges = edges
        self.results.bins = 0.5 * (edges[:-1] + edges[1:])

        # Need to know average area
        self.area = 0.0
        self.density_up = 0.0
        self.density_lo = 0.0
        self._maxrange = self.rdf_settings['range'][1]

        self.leaflets = AssignLeaflets(
            universe=self.u,
            lipid_sel=self.leaflets_lipid_selection,
            n_bins=self.n_bins_leaflet
        )
        self.leaflets.run(self.leaflets_start, self.leaflets_end, self.leaflets_step)

        self.g1_leaflets = self.leaflets.filter_leaflets(self.g1_leaflet_filter)
        self.g2_leaflets = self.leaflets.filter_leaflets(self.g2_leaflet_filter)

    def _single_frame(self):
        g1_pos = self.g1.positions 
        g2_pos = self.g2.positions 


        g1_up_mask = self.g1_leaflets[self.g1_local_resindices,self._frame_index] == 1
        g2_up_mask = self.g2_leaflets[self.g2_local_resindices,self._frame_index] == 1

        g1_lo_mask = self.g1_leaflets[self.g1_local_resindices,self._frame_index] == -1
        g2_lo_mask = self.g2_leaflets[self.g2_local_resindices,self._frame_index] == -1

        pairs_up, dist_up = distances.capped_distance(
            project_coords_array_on_plane(g1_pos[g1_up_mask], self.normal_vector),
            project_coords_array_on_plane(g2_pos[g2_up_mask], self.normal_vector),
            self._maxrange,
            box=self.u.dimensions
        )

        pairs_lo, dist_lo = distances.capped_distance(
            project_coords_array_on_plane(g1_pos[g1_lo_mask], self.normal_vector),
            project_coords_array_on_plane(g2_pos[g2_lo_mask], self.normal_vector),
            self._maxrange,
            box=self.u.dimensions
        )

        # Exclusions
        if self._exclusion_block is not None:
            idxA_up = pairs_up[:,0]//self._exclusion_block[0]
            idxB_up = pairs_up[:,1]//self._exclusion_block[1]
            mask_up = np.where(idxA_up != idxB_up)[0]
            dist_up = dist_up[mask_up]

            idxA_lo = pairs_lo[:,0]//self._exclusion_block[0]
            idxB_lo = pairs_lo[:,1]//self._exclusion_block[1]
            mask_lo = np.where(idxA_lo != idxB_lo)[0]
            dist_lo = dist_lo[mask_lo]

        count_up = np.histogram(dist_up, **self.rdf_settings)[0]
        self.results.count_up += count_up

        count_lo = np.histogram(dist_lo, **self.rdf_settings)[0]
        self.results.count_lo += count_lo

        n_g1_up = np.sum(self.g1_leaflets[:,self._frame_index] == 1)
        n_g2_up = np.sum(self.g2_leaflets[:,self._frame_index] == 1)
        N_up = n_g1_up * n_g2_up

        n_g1_lo = np.sum(self.g1_leaflets[:,self._frame_index] == -1)
        n_g2_lo = np.sum(self.g2_leaflets[:,self._frame_index] == -1)
        N_lo = n_g1_lo * n_g2_lo
        
        # Exclusions 
        if self._exclusion_block:
            xA, xB = self._exclusion_block
            nblocks_up = n_g1_up / xA
            N_up -= xA * xB * nblocks_up

            nblocks_lo = n_g1_lo / xA
            N_lo -= xA * xB * nblocks_lo

        self.area += self.u.dimensions[0] * self.u.dimensions[1] # This would need to be changed for anything other than z-axis
        self.density_up += N_up / (self.u.dimensions[0] * self.u.dimensions[1])
        self.density_lo += N_lo / (self.u.dimensions[0] * self.u.dimensions[1])


    def _conclude(self):

        # Area in each shell
        areas = np.power(self.results.edges, 2)
        area = np.pi * np.diff(areas)

        av_density_up = self.density_up / self.n_frames
        av_density_lo = self.density_lo / self.n_frames

        print(av_density_up, av_density_lo)

        rdf_up = self.results.count_up / (av_density_up * area * self.n_frames)
        rdf_lo = self.results.count_lo / (av_density_lo * area * self.n_frames)

        self.results.rdf_up = rdf_up
        self.results.rdf_lo = rdf_lo

        self.results.rdf = 0.5 * (rdf_up + rdf_lo)

