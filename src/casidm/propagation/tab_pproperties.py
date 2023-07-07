from particle import Particle
import numpy as np
import math
from casidm.data_structs.pdg_pid_map import PdgPidMap

class ParticlePropertiesParticle:
    """The class should provide
        get_ctau(pdg)
        get_mass(pdg)
        get_pmap(pdg)
    """ 
    
    def __init__(self):
        self.all_particles_dict = {p.pdgid: p for p in Particle.findall()}
        self.pdg2pid_map = {int(pdgid) : pid for pid, 
                               pdgid in enumerate(self.all_particles_dict)}
        self.pmap = PdgPidMap(self.pdg2pid_map)
        
        self._split_stable_decaying()
        

    def _split_stable_decaying(self):
        stable = []
        decaying = []
        decaying_ctau0 = {}
        for pdg in self.pdg2pid_map:
            ctau = self.all_particles_dict[pdg].ctau
            if (ctau is None) or math.isinf(ctau):
                stable.append(pdg)   
            else: 
                decaying.append(pdg)
                decaying_ctau0[pdg] = np.float64(ctau * 1e-1)     
         
        self.stable = np.array(stable)
        self.decaying = np.array(decaying)
        self.decaying_ctau0 = decaying_ctau0
            

    def get_ctau(self, pdg):
        """Returns c*tau, in cm"""
        ctau = self.all_particles_dict[pdg].ctau
        if (ctau is None) or math.isinf(ctau):
            return np.inf
        else:
            # 1e-1 is conversion factor from mm to cm
            return np.float64(ctau * 1e-1)
    
    def get_mass(self, pdg):
        """Returns mass in GeV"""
        
        try:
            if self.all_particles_dict[pdg].mass is None:
                return np.float64(0)
            else:
                # 1e-3 is conversion factor from MeV to GeV
                return np.float64(self.all_particles_dict[pdg].mass) * 1e-3
        except KeyError as ex:
            print(ex)
            # raise
    
                    

    def get_pmap(self):
        return self.pmap

class TabulatedParticleProperties:
    """Tabulate properties of particles as numpy arrays
    for fast access. The following properties are tabulated

    TabParticles.mass(pdg_array)
    TabParticles.ctau(pdg_array)
    """
    def __init__(self, particle_properties = ParticlePropertiesParticle()):
        self.get_ctau = particle_properties.get_ctau
        self.get_mass = particle_properties.get_mass
        self.pmap = particle_properties.get_pmap()
        self.tabulate_ctau()
        self.tabulate_mass()


    def tabulate_ctau(self):
        self._ctau_tab = np.zeros(len(self.pmap.pdg_pid), dtype=np.float64)
        for pid, pdg in enumerate(self.pmap.pdg_pid):
            self._ctau_tab[pid] = self.get_ctau(pdg)
            
    def tabulate_mass(self):
        self._mass_tab = np.zeros(len(self.pmap.pdg_pid), dtype=np.float64)
        for pid, pdg in enumerate(self.pmap.pdg_pid):
            self._mass_tab[pid] = self.get_mass(pdg)        


    def mass(self, pdg_array):
        try:
            mass_array = self._mass_tab[self.pmap.get_pids(pdg_array)]
        except IndexError:
            # It is assumed to be a rare case
            # print(f"Pdg_array = {pdg_array}")
            mass_array = np.frompyfunc(self.get_mass, 1, 1)(pdg_array).astype(
                "float64"
            )
        return mass_array

    def ctau(self, pdg_array):
        try:
            ctau_array = self._ctau_tab[self.pmap.get_pids(pdg_array)]
        except (IndexError):
            # It is assumed to be a rare case
            ctau_array = np.frompyfunc(self.get_ctau, 1, 1)(pdg_array).astype(
                "float64"
            )
        return ctau_array


if __name__ == "__main__":
    
    # print(len(pmap.pdg_pid))
    tpp = TabulatedParticleProperties()

    print(tpp.ctau([111, 11, 13, -13]))
    print(tpp.ctau([-1001112720, 1000902270, 1000721650]))

    print(tpp.mass(np.array([111, 12, 13])))
    print(tpp.mass([-1001112720, 1000902270, 1000721650]))
