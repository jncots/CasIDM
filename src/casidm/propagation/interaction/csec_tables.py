import MCEq
from MCEq.misc import normalize_hadronic_model_name
from MCEq.data import HDF5Backend, Interactions, InteractionCrossSections, Decays
from MCEq.particlemanager import ParticleManager
import numpy as np


class CSXdepthConversion:
    """Get conversion factor from cross section to xdepth"""
    
    def __init__(self, cs_unit = "mbarn"):
        self.mass_factor = self._get_mass_factor(cs_unit)
        self.average_amass_air = self._get_average_amass_air()
        self.cs_xdepth_air = self.mass_factor * self.average_amass_air
    
    
    def get_cs_xdepth_air(self):
        """Returns conversion factor from cross section to xdepth for air
        """
        return self.cs_xdepth_air
        
    
    def _get_mass_factor(self, cs_unit):
        """Returns mass factor = proton_mass [g]/cross_section_unit [cm^2]
        """
        area_in_cm2 = dict()
        area_in_cm2["cm2"] = 1
        area_in_cm2["mbarn"] = 1e-27
        proton_mass_g = 1.672621e-24
        return proton_mass_g / area_in_cm2[cs_unit]
            
    def _get_average_amass_air(self):
        """Returns average atomic mass of air
        """
        # Air composition is taken from 
        # https://www.engineeringtoolbox.com/molecular-mass-air-d_679.html
        # "Global mean water vapour is about 0.25% of the atmosphere by mass"
        # https://en.wikipedia.org/wiki/Water_vapor
        # It gives 0.2% of vapor by volume
        # air_composition = [(14, 2 * 0.78084),
        #                    (16, 2 * 0.20946),
        #                    (40, 0.00934),
        #                    (1, 2 * 0.002),
        #                    (16, 0.002)]
        
        
        air_composition = [(14, 0.78),
                           (16, 0.22)]
        # air_composition = [(14, 1)]
        
        total_fraction = 0 
        average_mass = 0
        for mass, fraction in air_composition:
            average_mass += mass * fraction
            total_fraction += fraction
        
        average_mass = average_mass/total_fraction
        
        return average_mass


class CrossSectionTableMCEq:
    def __init__(self, interaction_model="DPMJETIII191", cs_xdepth_conv = CSXdepthConversion()):
        self.interaction_model = interaction_model
        self.cs_xdepth_conv = cs_xdepth_conv
        
        self._initialize_mceq_data(interaction_model)
        self.set_energy_grid()
        
    def _initialize_mceq_data(self, interaction_model):
        """This is initialization of _int_cs and _pman required for 
        cross-section tabulation

        Args:
            interaction_model: e.g. "DPMJETIII191"
        """
        interaction_model = normalize_hadronic_model_name(interaction_model)

        medium = MCEq.config.interaction_medium
        _mceq_db = HDF5Backend(medium=medium)

        _interactions = Interactions(mceq_hdf_db=_mceq_db)
        _int_cs = InteractionCrossSections(mceq_hdf_db=_mceq_db)
        _int_cs.load(interaction_model)
        _decays = Decays(mceq_hdf_db=_mceq_db)

        _interactions.load(interaction_model)
        _decays.load(parent_list=_interactions.particles)
        _particle_list = _interactions.particles + _decays.particles
        self._pman = ParticleManager(
            _particle_list, 
            _mceq_db.energy_grid, 
            _int_cs, 
            medium
        )
        
        self._int_cs = _int_cs

    def set_energy_grid(self, *, emin=1e-1, emax=1e11, npoints=1000):
        pid_size = 0
        for p in self._pman.all_particles:
            if p.is_hadron:
                pid_size += 1
                
        self.energy_grid = np.geomspace(emin, emax, npoints, dtype='float64')
        self.sigma_tab = np.empty([pid_size, len(self.energy_grid)], dtype=np.float64)
        self.xdepth_tab = np.empty([pid_size, len(self.energy_grid)], dtype=np.float64)
        self._tabulate()
        
    def _tabulate(self):
        pid = 0
        self.pid_pdg = dict()  
        for p in self._pman.all_particles:
            if p.is_hadron:
                pdg = p.pdg_id[0]
                # print(f"Tabulate cross-section for {p.name}({pdg})")
                self.sigma_tab[pid, :] = np.interp(self.energy_grid, 
                                            self._int_cs.energy_grid.c + p.mass, 
                                            self._int_cs.get_cs(pdg, True))
                
                # If some points have -0.0 and other +0.0, then 
                # interpolation can give nan when xdepth is calculated
                # We set all zeros to +0.0, to get inf in xdepth
                self.sigma_tab[pid, self.sigma_tab[pid, :] == np.NZERO] = np.PZERO
                
                self.pid_pdg[pdg] = pid
                pid += 1
        with np.errstate(divide='ignore'):
            self.xdepth_tab[:] = np.divide(self.cs_xdepth_conv.get_cs_xdepth_air(), self.sigma_tab)
                 
            
    def get_sigma(self):
        return self.sigma_tab
    
    def get_xdepth(self):
        return self.xdepth_tab
        
    def get_energy_grid(self):
        return self.energy_grid
    
    def get_pid_pdg_dict(self):
        return self.pid_pdg

    def add_pdgs(self, pdg_pdg_map):
        """Adds allowed pdgs to pid_pdg dictionary.
        Added pdgs mapped to existing pids in dictionary

        Args:
            pdg_pdg_map (dict): Map new pdg (key) to existing
            pdg (value) in pid_pdg dictionary  
        """
        for pdg_from, pdg_to in pdg_pdg_map.items():
            self.pid_pdg[pdg_from] = self.pid_pdg[pdg_to]
     