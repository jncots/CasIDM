if __name__ == "__main__":
    from pathlib import Path
    import sys
    print(Path(__file__).parents[1])
    sys.path.append(str(Path(__file__).parents[1]))



from casidm.data_structs.particle_array import ParticleArray
# from casdata_structs.particle_array import ParticleArray
from casidm.propagation.tab_pproperties import TabulatedParticleProperties
from casidm.propagation.slant_depth.xdepth_on_table import XdepthOnTable
from casidm.data_structs.property_maps import tab_pprop
import numpy as np


class DecayXdepth:
    def __init__(self, *,
                 tab_particle_properties = TabulatedParticleProperties(),
                 xdepth_on_table = XdepthOnTable()):
        self.pp_tab = tab_particle_properties
        self.tab_pprop = tab_pprop
        self.xd_tab = xdepth_on_table

    def get_xdepth(self, pdg, energy, xdepth):
        
        # Check if particle doesn't decay      
        ctau = self.tab_pprop.ctau[pdg]      
        is_inf = ctau == np.inf
        inf_flag = is_inf.any()
        
        # Consider only decaying particles
        if inf_flag:
            not_inf = np.logical_not(is_inf)
            next_xdepth = np.empty_like(xdepth)
            # Stable particles decay at X = np.inf
            next_xdepth[is_inf] = np.inf
            
            ctau = ctau[not_inf]
            pdg = pdg[not_inf]
            energy = energy[not_inf]
            xdepth = xdepth[not_inf]
        
        mass = self.tab_pprop.mass[pdg]
        gamma = energy/mass
        bgamma = np.sqrt((gamma + 1) * (gamma - 1))
        rnd = -np.log(1 - np.random.rand(len(pdg)))
        length = rnd * bgamma * ctau
        
        if inf_flag:
            # Get result for decaying particles
            next_xdepth[not_inf] = self.xd_tab.add_len2x(xdepth, length)
            return next_xdepth
        else:
            return self.xd_tab.add_len2x(xdepth, length)


if __name__ == "__main__":
    particle_stack = ParticleArray(size=100)
    decay_xdepth = DecayXdepth()

    particle_stack.push(
        pid=np.array([111, 22, 111, 13, -13, 2212]),
        energy=np.array([2e10, 2e0, 2e10, 5e1, 1e0, 1e0]),
        xdepth=np.array([0, 0, 0, 0, 0, 0]),
    )

    print(particle_stack.xdepth[0 : len(particle_stack)])

    particle_stack.xdepth_decay[0 : len(particle_stack)] = decay_xdepth.get_xdepth(particle_stack.pid[0 : len(particle_stack)],
        particle_stack.energy[0 : len(particle_stack)],
        particle_stack.xdepth[0 : len(particle_stack)])

    print(particle_stack.xdepth_decay[0 : len(particle_stack)])
