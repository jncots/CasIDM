import chromo
from casidm.data_structs.particle_array import ParticleArray
import numpy as np


class HadronInteraction:
    def __init__(self, imodel, xdepth_getter):
        
        self.target = imodel.target
        self.generate_targets()
        self.event_generator = imodel.event_generator
        
        # Get a global PdgPidMap object
        # ToDo use the same object as in cross section tabulation 
        self.pmap = xdepth_getter.particle_properties.pmap
    
    def generate_targets(self):
        precalc_num = 100000
        components = np.array(self.target.components)
        tot_prob = np.zeros(len(self.target.fractions) + 1)
        tot_prob[1:] = np.cumsum(self.target.fractions)
        self.pids = components[np.digitize(np.random.random(precalc_num), tot_prob) - 1]
        self.pids_len = len(self.pids)
        self.pid_counter = -1
    
    def get_target(self):
        self.pid_counter += 1
        if self.pid_counter < self.pids_len:
            return int(self.pids[self.pid_counter])
        else:
            self.generate_targets()
            self.pid_counter += 1
            return int(self.pids[self.pid_counter])
            
            
    
    def run_event_generator(self, parents, children, failed_parents):
        
        children.clear()
        number_of_interactions = 0
        pvalid = parents.valid()
        pvalid.production_code[:] = 0
        
        for i in range(len(pvalid)):
            try:
                self.event_generator.kinematics = chromo.kinematics.FixedTarget(
                    pvalid.energy[i], int(pvalid.pid[i]), self.get_target()
                )
                
                if (self.event_generator.kinematics.ekin <= 2e0):
                    raise RuntimeError("Too low energy")
                
            except Exception as e:
                if "projectile" in str(e):
                    # projectile is not allowed
                    pvalid.production_code[i] = 111
                elif "center-of-mass" in str(e):
                    # center-of-mass energy  < minimum energy 10.0 GeV
                    pvalid.production_code[i] = 222
                else:
                    pvalid.production_code[i] = 333
                
                continue    
            
            
            event = next(self.event_generator(1)).final_state()                
            number_of_interactions += 1
            generation_num = pvalid.generation_num[i] + 1
                        
            # Filter out only known pdgs, because some models
            # (such as DpmjetIII) sometimes produce
            # particles with unknown pdgs (unknown to 
            # `Particle` package which is 
            # used for initialization)
            # This is requires copy which slightly reduce speed
            # event = event[self.pmap.valid_pdg_indices(event.pid)]


            p1 = children._len
            p2 = p1 + len(event.pid)
            children._adjust_capacity(p2)
            
            children.pid[p1:p2] = event.pid
            children.energy[p1:p2] = event.en
            children.xdepth[p1:p2] = pvalid.xdepth_inter[i]
            children.generation_num[p1:p2] = generation_num
            children.parent_id[p1:p2] = pvalid.id[i]
            children.production_code[p1:p2] = 777
            children._len = p2
            
        failed_parents.refill(pvalid[np.where(pvalid.production_code > 0)])
        
        return number_of_interactions
        

if __name__ == "__main__":
    
    hint = HadronInteraction()
    
    parents = ParticleArray()
    children = ParticleArray()
    failed_parents = ParticleArray()
    
    parents.push(pid = 2212, energy = 1e5, xdepth = 0, xdepth_inter = 5, generation_num = 0)
    
    hint.run_event_generator(parents, children, failed_parents)
    parents1 = children.valid().copy()
    hint.run_event_generator(parents1, children, failed_parents)
    # hint.run_event_generator(ch)
    # ch = hint.get_children().valid().copy()
    # hint.run_event_generator(ch)
    # ch = hint.get_children().valid()
    ch = children.valid()
    fp = failed_parents.valid()
    print("pid = ", ch.pid)
    print("energy = ", ch.energy)
    print("xdepth = ", ch.xdepth)
    print("generation = ", ch.generation_num)
    print(len(ch))
    
    print("failed pid = ", fp.pid)
    print("failed energy = ", fp.energy)
    print("failed xdepth = ", fp.xdepth)
    print("failed generation = ", fp.generation_num)
    print(len(fp))
    