import numpy as np
from particle import Particle

def unique_pdg_list(pdgs):
    res = list(set(pdgs))
    res.sort(key=lambda x: (abs(x), x > 0))
    return res

class PdgPidMap:
    def __init__(self, pid_pdg_dict, max_pdg = 6000):
        self.max_pdg = max_pdg
        self.pid_pdg_dict = pid_pdg_dict
        self._build_maps() 
                         
    def _build_maps(self):
        
        self.none_value = -2147483640
        self.known_pdg_ids = np.array(unique_pdg_list(self.pid_pdg_dict.keys()))
        self.known_pid_ids = np.array(unique_pdg_list(self.pid_pdg_dict.values()))
        
        max_pdg_in_dict = max([abs(pdg) for pdg in self.pid_pdg_dict])
        if  max_pdg_in_dict <= self.max_pdg:
            total_map = True
            max_pdg_map = max_pdg_in_dict
        else:
            total_map = False
            max_pdg_map = self.max_pdg
                
        max_pid_in_dict = max([abs(pid) for pid in self.pid_pdg_dict.values()])          
        pdg_pid = np.full(max_pid_in_dict + 1, self.none_value, dtype = np.int32)                 
        pid_pdg = np.full(2 * max_pdg_map + 1, self.none_value, dtype=np.int32)
        
        for pdg, pid in self.pid_pdg_dict.items():
            pdg_pid[pid] = pdg
            if abs(pdg) <= max_pdg_map:
                pid_pdg[pdg] = pid

        self.total_map = total_map
        self.max_pdg_map = max_pdg_map
        self.pdg_pid = pdg_pid
        self.pid_pdg = pid_pdg
        self.max_pid = max(pid_pdg)
    
    
    def valid_pid_indices(self, pids):
        return np.where(np.isin(pids, self.known_pid_ids))[0]       
    
    def valid_pdg_indices(self, pdgs):
        return np.where(np.isin(pdgs, self.known_pdg_ids))[0]
    
    def get_pdgs(self, pids):
        try:
            return self.pdg_pid[pids]
        except:
            # Filter out not valid ids
            pids_np = np.array(pids)
            valid_ids = self.valid_pid_indices(pids_np)
            final_pdgs = np.full(len(pids_np), self.none_value, dtype=np.int32)
            final_pdgs[valid_ids] = self.pdg_pid[pids_np[valid_ids]]
            return final_pdgs
            
                
    def get_pids(self, pdgs):
        try:
            return self.pid_pdg[pdgs]
        except:
            try:
                return self._get_pids_full(pdgs)
            except:
                # Filter out not valid ids
                pdgs_np = np.array(pdgs)
                valid_ids = self.valid_pdg_indices(pdgs_np)
                final_pids = np.full(len(pdgs), self.none_value, dtype=np.int32)
                final_pids[valid_ids] = self.get_pids(pdgs_np[valid_ids])
                return final_pids
        
    def _get_pids_full(self, pdgs):
        return np.array([self.pid_pdg_dict[pdg] for pdg in pdgs], 
                        dtype = np.int32)
        
        
class PdgLists:
    def __init__(self):
        
        all_particles_dict = {p.pdgid: p for p in Particle.findall()}
        pdgs_6000 = [] 
        for p in all_particles_dict:
            pdg = int(p)
            if 10 < abs(pdg) < 6000:
                pdgs_6000.append(pdg)
                
        pdgs_6000.sort(key=lambda x: (abs(x), x > 0))
        self.pdgs_below_abs6000 = np.array(pdgs_6000, dtype = np.int32)
        
        
        
        
        self.sibyll_valid_pdgs = ([113, 130, 211, 310, 321, 411, 421, 431,
                        2112, 2212, 3112, 3122, 3212, 3222,
                        3312, 3322, 4122, 4132, 4232, 4332])

        self.longer_pi0 = ([130, 211, 310, 321, 411, 421, 431,
                511, 521, 531, 541, 2112, 2212, 3112,
                3122, 3222, 3312, 3322, 3334, 4122,
                4132, 4232, 4332, 5122, 5132, 5232, 5332])

        self.longer_30ps = ([130, 211, 310, 321, 2112, 2212, 
                        3112, 3122, 3222, 3312, 3322, 3334])

        self.final_pdgs = np.array([-11, 11, 12, -13, 13, 14, 16, 22], dtype = np.int32)
        self.finals = ([11, 12, 13, 14, 16, 22])
        

        self.mceq_hadrons = np.array([111, 130, -211, 211, 310,
                                       -321, 321, -2112, 2112, 
                                       -2212, 2212, -3122, 3122,
                                       ], dtype = np.int32)
            
        self.mceq_finals = np.array([-11, 11, 12, -13, 13, 14, 16, 22], dtype = np.int32)
                
        self.mceq_particles = np.array([-11, 11, -12, 12, -13, 13, -14, 14, -16, 16, 22, 
                                        111, 130, -211, 211, 310, -321, 321, 
                                        -411, 411, -421, 421, -431, 431, 
                                        -2112, 2112, -2212, 2212, -3122, 3122
                                       ], dtype = np.int32)
        
        
        self.not_mceq_below_abs6000 = (self.pdgs_below_abs6000[np.where(np.logical_not(
            np.isin(self.pdgs_below_abs6000, self.mceq_particles)))[0]])      
        
        self.pid_pdg_mceq = ({111: 0, 130: 1, -211: 2, 211: 3, 
                    310: 4, -321: 5, 321: 6, -411: 7, 
                    411: 8, -421: 9, 421: 10, -431: 11, 
                    431: 12, -2112: 13, 2112: 14, -2212: 15, 
                    2212: 16, -3122: 17, 3122: 18})

        self._set_longer_pi0_to_mceq()
        
        self.leptons_mceq = np.array([-11, 11, -12, 12, -13, 13, -14, 14, -16, 16, 22])
        self.leptons_stable_mceq = np.array([-11, 11, -12, 12, -14,  14, -16,  16, 22])
        self.leptons_decay_mceq = np.array([-13,  13])
        
        self.hadrons_mceq = np.array([111, 130, -211, 211, 310, -321, 321, -411,   
                                      411, -421, 421, -431, 431, 
                                      -2112, 2112, -2212, 2212, -3122, 3122])
        
        self.hadrons_mix_mceq = np.array([111, 130, -211, 211, 310, -321, 321, -411,   
                                      411, -421, 421, -431, 431, 
                                      -2112, 2112, -3122, 3122])
        
        self.hadrons_stable_mceq = np.array([-2212, 2212])
        
        self.hadron_emix_map = {111: 1122018454.3019652, 
                                130: 22.38721138568341, 
                                -211: 7.079457843841384, 
                                211: 8.91250938133746, 
                                310: 8912.509381337459, 
                                -321: 70.79457843841384, 
                                321: 89.12509381337459, 
                                -411: 2238721.138568342, 
                                411: 2238721.138568342, 
                                -421: 4466835.921509635, 
                                421: 4466835.921509635, 
                                -431: 4466835.921509635, 
                                431: 4466835.921509635, 
                                -2112: 0.11220184543019636, 
                                2112: 0.11220184543019636, 
                                -3122: 5623.4132519034965, 
                                3122: 5623.4132519034965}
        
        hmix_map = PdgPidMap({pdg : pid for pid, pdg in 
                              enumerate(self.hadrons_mix_mceq)})
        self.hadron_emix = np.zeros_like(hmix_map.pid_pdg, dtype=np.float64)
        for pdg, emix in self.hadron_emix_map.items():
            self.hadron_emix[pdg] = emix
    
    def _set_longer_pi0_to_mceq(self):    
        self.longer_pi0_to_mceq = ({-511: -411, 511: 411, -521: -421, 521: 421,
                        -531: -431, 531: 431, -541: -431, 541: 431})
        
        for pdg in self.longer_pi0:
            if pdg > 3000:
                self.longer_pi0_to_mceq[-pdg] = -3122
                self.longer_pi0_to_mceq[pdg] = 3122
            
            