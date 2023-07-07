if __name__ == "__main__":
    from pathlib import Path
    import sys
    print(Path(__file__).parents[2])
    sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
from casidm.data_structs.pdg_pid_map import PdgPidMap

                          
class CrossSectionOnTable:
    def __init__(self, cross_section_table):
        # Cross section table
        self.sigma_tab = cross_section_table.get_sigma()
        # Xdepth table
        self.xdepth_tab = cross_section_table.get_xdepth()
        # Energy grid
        self.energy_grid = cross_section_table.get_energy_grid()
        # Pdg to pid maps
        self.pmap = PdgPidMap(cross_section_table.get_pid_pdg_dict())
    
    def get_cross_section(self, pdg, energy, sigma):
        """Fills sigma array with cross section for particles with pdg and energy

        Args:
            pdg (np.array): array of pdgs
            energy (np.array): array of energies
            sigma (np.array): output array with cross section
        """
        pid_array = self.pmap.get_pids(pdg)
        for pid in set(pid_array):
            pid_slice = np.where(pid_array == pid)[0]
            sigma[pid_slice] = np.interp(energy[pid_slice], self.energy_grid, self.sigma_tab[pid,:])
            
    def get_mean_xdepth(self, pdg, energy):
        """Get xdepth array with mean xdepth for particles with pdg and energy

        Args:
            pdg (np.array): array of pdgs
            energy (np.array): array of energies
            xdepth (np.array): output array with xdepth
        """
        xdepth = np.full(len(pdg), np.inf, dtype=np.float64)
        # Choose only pdgs for which cross sections are tabulated
        valid_inds = self.pmap.valid_pdg_indices(pdg)
        pid_array = self.pmap.get_pids(pdg)
        valid_pids_set = set(self.pmap.get_pids(pdg[valid_inds]))
        for pid in valid_pids_set:
            pid_slice = np.where(pid_array == pid)[0]
            xdepth[pid_slice] = np.interp(energy[pid_slice], self.energy_grid, self.xdepth_tab[pid,:])
                    
        return xdepth
            
    def get_xdepth(self, pdg, energy):
        """Return xdepth array with (random) xdepth for next interaction 
        for particles with pdg and energy

        Args:
            pdg (np.array): array of pdgs
            energy (np.array): array of energies
            xdepth (np.array): output array with xdepth
        """
        mean_xdepth = self.get_mean_xdepth(pdg, energy)        
        rnd = -np.log(1 - np.random.rand(len(pdg)))
        return mean_xdepth * rnd  
                 

 
if __name__ == "__main__":
    from csec_tables import CrossSectionTableMCEq
    from data_structs.pdg_pid_map import PdgPidMap, PdgLists
    
    cs_table = CrossSectionTableMCEq()
    # cs_table.add_pdgs(PdgLists().longer_pi0_to_mceq)
    
    csec = CrossSectionOnTable(cs_table)
    
    

    # pdg_list = np.array([111, 111, 111, 
    #                     111, -211, 
    #                     111, 111, 2212, 111], dtype=np.int32)
    # pdg_list = np.array([13, -13, 11, -11, 22, 111, -211], dtype=np.int32)
    # energy_list = np.array([5e3, 5e2, 4e6, 4e3, 2e5, 2e5, 4e3, 5e2, 2e5], dtype=np.float64) 
    
    pdg_list = np.array([211, -211, 2212], dtype=np.int32)
    energy_list = np.array([1e-1, 1e4, 1e3], dtype=np.float64)   

    print(csec.get_mean_xdepth(pdg_list, energy_list))
    print(csec.get_xdepth(pdg_list, energy_list))