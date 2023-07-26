from casidm.propagation.decay_xdepth import DecayXdepth
from casidm.propagation.interaction.csec_tables import CrossSectionTableMCEq, CSXdepthConversion
from casidm.propagation.interaction.csec_on_table import CrossSectionOnTable

from casidm.data_structs.pdg_pid_map import PdgLists
from casidm.data_structs.particle_array import FilterCode
from casidm.propagation.slant_depth.xdepth_on_table import XdepthOnTable
from casidm.propagation.slant_depth.xdepth_conversion import XdepthConversion
from MCEq.geometry.density_profiles import CorsikaAtmosphere
from casidm.propagation.tab_pproperties import TabulatedParticleProperties, ParticlePropertiesParticle
import numpy as np


class NextDecayXdepth:
    def __init__(self, *, xdepth_on_table, 
                 particle_properties = ParticlePropertiesParticle()):
        tab_particle_properties = TabulatedParticleProperties(particle_properties=particle_properties)
        self.decay_xdepth = DecayXdepth(tab_particle_properties=tab_particle_properties,
                                        xdepth_on_table=xdepth_on_table)
        self.max_xdepth = xdepth_on_table.xdepth_conversion.get_max_xdepth()
        self._stop_xdepth = None
    
    def set_stop_xdepth(self, stop_xdepth):
        self._stop_xdepth = stop_xdepth        
         
    def get_xdepth(self, pstack):
        """Set xdepth_decay and filter_code for pstack[0:len(pstack)]
        """ 
        
        if pstack.slice_idx is None:
            # Almost always this branch is triggered
            pvalid = pstack.valid()
            pvalid.xdepth_decay[:] = (self.decay_xdepth
                                        .get_xdepth(pdg = pvalid.pid, 
                                                    energy = pvalid.energy, 
                                                    xdepth = pvalid.xdepth))
                    
            if self._stop_xdepth is not None:
                pvalid.xdepth_decay[pvalid.xdepth_decay >= self._stop_xdepth] = self._stop_xdepth
            pvalid.filter_code[:] = FilterCode.XD_DECAY_ON.value
        
        else:
            # This is for decay driver specifically
           
            pstack.xdepth_decay[pstack.slice_idx] = (self.decay_xdepth
                                        .get_xdepth(pdg = pstack.pid[pstack.slice_idx], 
                                                    energy = pstack.energy[pstack.slice_idx], 
                                                    xdepth = pstack.xdepth[pstack.slice_idx])) 

            pstack.filter_code[pstack.slice_idx] = FilterCode.XD_DECAY_ON.value
            
        
class NextInterXdepth:
    def __init__(self, *, xdepth_on_table):        
        cs_xdepth_conv = CSXdepthConversion()
        cs_table = CrossSectionTableMCEq(interaction_model="DPMJETIII191",
                                         cs_xdepth_conv = cs_xdepth_conv)
        # cs_table.add_pdgs(PdgLists().longer_pi0_to_mceq)
        self.inter_xdepth = CrossSectionOnTable(cs_table)
        self.max_xdepth = xdepth_on_table.xdepth_conversion.get_max_xdepth()
        self.known_pdg_ids = self.inter_xdepth.pmap.known_pdg_ids
        self._stop_xdepth = None
    
    
    def set_stop_xdepth(self, stop_xdepth):
        self._stop_xdepth = stop_xdepth    
         
    def get_xdepth(self, pstack):
        """Set xdepth_inter for pstack[0:len(pstack)]
        
        Make sure that pstack._len is correct
        """       
        pvalid = pstack.valid()
        result_with_infs = (self.inter_xdepth
                            .get_xdepth(pdg = pvalid.pid, 
                                        energy = pvalid.energy) + pvalid.xdepth)
        
        # pstack.xdepth_decay[np.where(result_with_infs >= self.max_xdepth)] = np.inf
        
        if self._stop_xdepth is not None:             
            pvalid.xdepth_inter[:] = np.where(result_with_infs >= self._stop_xdepth, 
                                               self._stop_xdepth, result_with_infs)
        else:
            pvalid.xdepth_inter[:] = np.where(result_with_infs >= self.max_xdepth, 
                                               self.max_xdepth, result_with_infs)
                
        

class DefaultXdepthGetter:
    def __init__(self, theta_deg = 0, mode="both"):
        self.atmosphere = CorsikaAtmosphere("USStd", None)
        self.xdepth_conversion =  XdepthConversion(atmosphere = self.atmosphere)
        self.xdepth_conversion.set_theta(theta_deg)
        self.max_xdepth = self.xdepth_conversion.get_max_xdepth()
        self.xdepth_on_table = XdepthOnTable(xdepth_conversion = self.xdepth_conversion, npoints=1000)
        self.particle_properties = ParticlePropertiesParticle()

        if mode == "both":
            self.next_decay = NextDecayXdepth(xdepth_on_table=self.xdepth_on_table,
                                              particle_properties = self.particle_properties)
            self.next_inter = NextInterXdepth(xdepth_on_table=self.xdepth_on_table)
        elif mode == "decay":
            self.next_decay = NextDecayXdepth(xdepth_on_table=self.xdepth_on_table)
        elif mode == "inter":
            self.next_inter = NextInterXdepth(xdepth_on_table=self.xdepth_on_table)
        else:
            raise ValueError("this should not happen")    
                
    
    def set_stop_xdepth(self, stop_xdepth):
        self.next_decay.set_stop_xdepth(stop_xdepth)
        self.next_inter.set_stop_xdepth(stop_xdepth)
                    
    
    def get_decay_xdepth(self, pstack):
        return self.next_decay.get_xdepth(pstack)
    
    def get_inter_xdepth(self, pstack):
        return self.next_inter.get_xdepth(pstack)
    
# default_xdepth_getter = DefaultXdepthGetter(mode="decay")
# default_xdepth_getter = DefaultXdepthGetter()
    

        
if __name__ == "__main__":
    from particle_array import ParticleArray
    import numpy as np
    
    
    # atmosphere = CorsikaAtmosphere("SouthPole", "December")
    # xdepth_conversion =  XdepthConversion(atmosphere = atmosphere)
    # xdepth_conversion.set_theta(30)
    # xdepth_on_table = XdepthOnTable(xdepth_conversion = xdepth_conversion, npoints=1000)
    
    # next_decay = NextDecayXdepth(xdepth_on_table=xdepth_on_table)
    # next_inter = NextInterXdepth(xdepth_on_table=xdepth_on_table)
    
    xdepth_getter = DefaultXdepthGetter()
    xdepth_getter.set_stop_xdepth(500)
    
    pstack = ParticleArray(size=100)
    # pstack.push(
    #     pid=np.array([111, 22, 111, 13, -13, 2212]),
    #     energy=np.array([2e10, 2e0, 2e10, 1e0, 1e0, 1e0]),
    #     xdepth=np.array([100, 56, 100, 98, 56, 500]),
    # )
    
    pstack.push(
        pid=np.array([-13, 13, -11, 11]),
        energy=np.array([1e-1, 1e-1, 1e3, 1e3]),
        xdepth=np.array([1e1, 1e1, 1e1, 1e1]),
    )
    
    print(pstack.xdepth_decay[0:len(pstack)])
    print(pstack.xdepth_inter[0:len(pstack)])
    
    xdepth_getter.get_decay_xdepth(pstack)
    xdepth_getter.get_inter_xdepth(pstack)
    
    print("After")
    print("Decay", pstack.valid().xdepth_decay)
    print("Inter", pstack.valid().xdepth_inter)
    
    
         