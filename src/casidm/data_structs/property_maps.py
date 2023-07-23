import numpy as np
import math
import particle


def unique_sorted_ids(ids):
    """np.arrays of unique ids sorted by abs value
    with negative value first, e.g.
    -11, 11, -12, 12, ...
    """
    uids = np.unique(np.fromiter(ids, dtype=np.int32))
    return uids[np.argsort(2*np.abs(uids) - (uids < 0))]

def search_min_modulo(array):
    m_min = 1
    m_max = np.max(np.abs(array))
    for i in range(m_min, m_max):
        unq = np.unique(array % i)
        if len(unq)==len(array):
            return i
    return 0

    

def pdgmap2numpy(val_dict, max_pdg = 10000000):
    all_pdgs = unique_sorted_ids(val_dict)
    if max_pdg is None:
        known_pdgs = all_pdgs
    else:
        known_pdgs = all_pdgs[np.abs(all_pdgs) < max_pdg]
        
    values = []
    for i in known_pdgs:
        values.append(val_dict[i])
    
    values = np.array(values)
    if isinstance(values[0], int):
        values = values.astype(np.int32)
    elif isinstance(values[0], float):
        values = values.astype(np.float64)  
    return known_pdgs, values


def modulo_map(known_pdgs, known_case = None):
    cached_cases = {"only_particles": 5866, "all" : 66765}
    
    try:        
        modulus = cached_cases[known_case]
    except:
        modulus = search_min_modulo(known_pdgs)
    key_hash = known_pdgs % modulus
    return modulus, key_hash

def get_map(keys, values, modulus):
    map_array = np.full(modulus, np.nan, dtype=values.dtype)
    key_map = keys%modulus
    map_array[key_map] = values
    return map_array


            
def pick_none_int(values):
    clear_ints = np.array([333, 555, 777, 999, np.iinfo(values.dtype).min])
    for nval in clear_ints:
        if np.all(nval != values):
            return nval
    raise ValueError("No suitable ints for none value in clear_ints\n"
                     "Add another int(notable) to clear_ints to continue")
    
    
def pick_none_value(values):
    none_value = np.nan
    if np.issubdtype(values.dtype, np.integer):
        none_value = pick_none_int(values)
        
    if np.issubdtype(values.dtype, np.bool_):
        none_value = False   
    
    return none_value


def calc_vmap_size(keys):
    max_key = np.max(keys)
    min_key = np.min(keys)
    
    positive_size = 0
    if max_key > 0:
        positive_size = max_key + 1
    
    negative_size = 0
    if -min_key > 0:
        negative_size = -min_key
    
    return positive_size + negative_size
            
            
def value_map(key_hash, values):
    vmap_size = calc_vmap_size(key_hash)
    none_value = pick_none_value(values)
    vmap = np.full(vmap_size, none_value, dtype=values.dtype)
    vmap[key_hash] = values
    return vmap, none_value

 
class PropertyMapMod:
    def __init__(self, val_dict, max_pdg=None, special_case=None):
        self.val_dict = val_dict
        
        if special_case is not None:
            cached_cases = {"only_particles": 10000000, "all" : None}
            max_pdg = cached_cases[special_case]
        self.known_pdgs, self.values = pdgmap2numpy(self.val_dict, max_pdg)
        
        self.modulus, self.key_hash = modulo_map(self.known_pdgs, special_case)
        self.vmap, self.none_value = value_map(self.key_hash, self.values)
        self.kmap, self.none_key = value_map(self.key_hash, self.known_pdgs)
        
    def __getitem__(self, pdgs):
        return self.vmap[pdgs % self.modulus]
    
class PropertyMapInd:
    def __init__(self, val_dict, max_pdg):
        self.val_dict = val_dict
        self.known_pdgs, self.values = pdgmap2numpy(self.val_dict, max_pdg)
        
        self.vmap, self.none_value = value_map(self.known_pdgs, self.values)
        self.kmap, self.none_key = value_map(self.known_pdgs, self.known_pdgs)
        
    def __getitem__(self, pdgs):
        return self.vmap[pdgs]    


class PropertyMap:
    def __init__(self, val_dict, only_particles = True):
        self.short_map = PropertyMapInd(val_dict, max_pdg = 6000)
        self.short_vmap = self.short_map.vmap
        
        if only_particles:
            self.full_map = PropertyMapMod(val_dict, max_pdg = None, 
                                    special_case = "only_particles")
        else:
            self.full_map = PropertyMapMod(val_dict, max_pdg = None, 
                                    special_case = "all")
        
        self.modulus = self.full_map.modulus
        self.full_vmap = self.full_map.vmap
        self.full_max = np.max(np.abs(self.full_map.known_pdgs))   
        
    def __getitem__(self, pdgs):
        try:
            return self.short_vmap[pdgs]
        except IndexError:
            if np.max(np.abs(pdgs)) > self.full_max:
                out_range = pdgs[np.max(np.abs(pdgs)) > self.full_max]
                raise IndexError(f"pdgs out of range: {out_range}")
            return self.full_vmap[pdgs % self.modulus]
                    


def mass_dict():
    """Returns mass in GeV"""
    masses = dict()
    for p in particle.Particle.findall():
        if p.mass is None:
            mass = 0.0
        
        mass = p.mass*1e-3
        masses[int(p.pdgid)] = mass
    return masses


def ctau_dict():
    """Returns c*tau, in cm"""
    ctaus = dict()
    for p in particle.Particle.findall():
        ctau = p.ctau
        if  (ctau is None) or math.isinf(ctau):
           ctau = np.inf
        
        # 1e-1 is conversion factor from mm to cm
        ctau = np.float64(ctau * 1e-1)
        ctaus[int(p.pdgid)] = ctau
    return ctaus


def pdg_set_dict(pdgs):
    
    pdgs = unique_sorted_ids(pdgs)
    pdgs_dict = dict()
    
    for p in particle.Particle.findall():
        pdg = int(p.pdgid)
        
        if np.isin(pdg, pdgs):
            pdgs_dict[pdg] = True
        else:
            pdgs_dict[pdg] = False
                   
    return pdgs_dict

def get_pdg_set(pdgs, only_particles=True):
    return PropertyMap(pdg_set_dict(pdgs), only_particles)


class TabulatedProp:
    def __init__(self):
        self.mass = PropertyMap(mass_dict())
        self.ctau = PropertyMap(ctau_dict())
    


if __name__ == "__main__": 
    # mass = PropertyMap(get_mass_dict(), False)
    # print(mass[np.array([111, 111, 22, 14, -14, 1, 777, 325, 343243])])
    my_set = get_pdg_set([111, 111, 22, 14, -14, 1, 777])
    print(my_set[[113, 112, 22, 14, -14, 1, 11, 12]])
    
                    
            
            