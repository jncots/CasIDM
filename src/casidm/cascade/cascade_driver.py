import numpy as np


from casidm.data_structs.particle_array import ParticleArray, FilterCode
from casidm.data_structs.pdg_pid_map import PdgLists, PdgPidMap, unique_sorted_ids
from casidm.data_structs.id_generator import IdGenerator

from casidm.propagation.particle_xdepths import DefaultXdepthGetter

from casidm.process.hadron_inter import HadronInteraction
from casidm.process.decay_driver import DecayDriver
from casidm.utils.utils import suppress_std_streams

from tqdm import tqdm
import numpy as np
import time



class InteractionModel:
    def __init__(self, model, initial_kinematics, target):
        with suppress_std_streams():
            self.target = target
            self.event_generator = model(initial_kinematics)
        
        

class CascadeDriver:
    def __init__(self, imodel):
        self.imodel = imodel
        self.id_generator = IdGenerator()
        self.pdg_lists = PdgLists()
        
        
        self.final_decay_stack = ParticleArray()
        self.final_stack = ParticleArray()
        self.archival_stack = ParticleArray()
        
        self.working_stack = ParticleArray()
        self.spare_working_stack = ParticleArray()
        self.below_threshold_stack = ParticleArray()
        self.above_threshold_stack = ParticleArray()
        
        self.propagating_stack = ParticleArray()
        self.decay_stack = ParticleArray()
        self.inter_stack = ParticleArray()
        self.rejection_stack = ParticleArray()
        self.children_stack = ParticleArray()
    
    
    def set_zenith_angle(self, zenith_angle):
        self.xdepth_getter = DefaultXdepthGetter(zenith_angle)
        self.decay_driver = DecayDriver(self.xdepth_getter)
        self.hadron_interaction = HadronInteraction(self.imodel, self.xdepth_getter)
        
        self.interacting_pdgs = self.xdepth_getter.next_inter.known_pdg_ids
        self.phys_stable_pdgs = self.xdepth_getter.particle_properties.stable
        self.phys_decaying_pdgs = self.xdepth_getter.particle_properties.decaying
        
        
    def set_decaying_pdgs(self, pdgs_categories):
        
        self.mceq_mixed_pdgs = pdgs_categories["mixed"]["pdg_ids"]
        self.mceq_mixed_energy = pdgs_categories["mixed"]["etot_mix"]
        self.mceq_mixed_map = PdgPidMap({pdg: pid for pid, pdg in enumerate(self.mceq_mixed_pdgs)})

        self.mceq_final_pdgs = unique_sorted_ids(pdgs_categories["final"])
        self.mceq_only_interacting_pdgs = unique_sorted_ids(pdgs_categories["only_interacting"])
        self.mceq_only_decaying_pdgs = unique_sorted_ids(pdgs_categories["only_decaying"])
        self.mceq_resonance_pdgs = unique_sorted_ids(pdgs_categories["resonance"])
        
        self.unconditionally_final_pdgs = unique_sorted_ids(np.concatenate([self.mceq_final_pdgs,
                                                           self.mceq_only_decaying_pdgs]))
        
        self.interacting_decaying_pdgs = unique_sorted_ids(np.concatenate([self.mceq_mixed_pdgs,
                                                                        self.mceq_only_interacting_pdgs]))
                
        is_final_pdgs = np.logical_not(np.isin(self.pdg_lists.mceq_particles, 
                                                                self.mceq_resonance_pdgs))
        self.final_pdgs = unique_sorted_ids(self.pdg_lists.mceq_particles[is_final_pdgs])
        
         
        # By default all particles that can physically decay will decay in pythia
        # pdgs in stable_pdgs will not be decayed in pythia
        self.decay_driver.set_decaying_pdgs(decaying_pdgs=self.phys_decaying_pdgs)  
        
    
    def get_mix_energy(self, pdg):
        return self.mceq_mixed_energy[self.mceq_mixed_map.get_pids(pdg)]
            
        
    def simulation_parameters(self, *, pdg, energy, 
                              threshold_energy,
                              zenith_angle,
                              xdepth = 0,
                              stop_height = 0,
                              accumulate_runs = False,
                              reset_ids = False,
                              pdgs_categories):
        
        self.cought_pions = None    
        self.initial_pdg = pdg
        self.initial_energy = energy
        self.threshold_energy = threshold_energy
        self.initial_xdepth = xdepth
        
        self.set_zenith_angle(zenith_angle)
        self.set_decaying_pdgs(pdgs_categories)
        
        self.stop_xdepth = self.xdepth_getter.xdepth_conversion.convert_h2x(stop_height * 1e5)
        self.xdepth_getter.set_stop_xdepth(self.stop_xdepth)
        # print(f"stop depth = {self.stop_xdepth}")
        # print(f"Interacting pdgs = {self.interacting_pdgs}"
        #       f"Number = {len(self.interacting_pdgs)}")
        
        if reset_ids:
            self.id_generator = IdGenerator()
        
        self.initial_run = True    
        self.accumulate_runs = accumulate_runs           
    
    
    def run(self, nshowers = 1):
        with suppress_std_streams(suppress_stderr=False):
            for _ in tqdm(range(nshowers), total = nshowers):
                self.run_once()
    
    def run_once(self):
        
        self.working_stack.clear()
        self.decay_stack.clear()
                
        if self.initial_run:
            self.final_stack.clear()
            self.final_decay_stack.clear()
            self.archival_stack.clear()
            self.number_of_decays = 0
            self.number_of_interactions = 0
            self.loop_execution_time = 0
            self.runs_number = 0
            
            if self.accumulate_runs:
                self.initial_run = False  
            
        
        self.working_stack.push(pid = self.initial_pdg, 
                         energy = self.initial_energy, 
                         xdepth = self.initial_xdepth,
                         generation_num = 0)
        
        self.id_generator.generate_ids(self.working_stack.valid().id)
        
        iloop = 1
        
        
        start_time = time.time()        
        while len(self.working_stack) > 0:
            
            # print(f"\r{iloop} Number of inter = {self.number_of_interactions}"
            #       f" number of decays = {self.number_of_decays}")
            
            # print(f"{iloop} Working stack = {len(self.working_stack)}")
            self.filter_uncond_final()
            self.filter_by_threshold_energy()
            self.handle_below_threshold()
            self.handle_above_threshold()
            self.filter_by_slant_depth()       
            self.run_hadron_interactions()
            if len(self.working_stack) == 0:
                self.run_particle_decay()
            
            iloop += 1
        
        self.run_final_forced_decay()
        
        self.loop_execution_time += time.time() - start_time
        self.runs_number += 1
    
    
    def set_decay_at_place(self, wstack):
        wstack.xdepth_decay[:] = wstack.xdepth
        wstack.filter_code[:] = FilterCode.XD_DECAY_ON.value
        return wstack
    
    def filter_uncond_final(self):
        wstack = self.working_stack.valid()
        is_uncond_final = np.isin(wstack.pid, self.unconditionally_final_pdgs)
        is_not_uncond_final = np.logical_not(is_uncond_final)
        self.final_stack.append(wstack[is_uncond_final])
        
        self.spare_working_stack.clear()
        self.spare_working_stack.append(wstack[is_not_uncond_final])
    
    def filter_by_threshold_energy(self):
        wstack = self.spare_working_stack.valid()
        
        is_above_threshold = wstack.energy > self.threshold_energy
        is_below_threshold = np.logical_not(is_above_threshold)
        
        self.above_threshold_stack.clear()
        self.above_threshold_stack.append(wstack[is_above_threshold])
        
        self.below_threshold_stack.clear()
        self.below_threshold_stack.append(wstack[is_below_threshold])
        
    def handle_below_threshold(self):
        wstack = self.below_threshold_stack.valid()   
        
        is_mceq = np.isin(wstack.pid, self.pdg_lists.mceq_particles)
        is_not_mceq = np.logical_not(is_mceq)
        
        self.final_decay_stack.append(wstack[is_not_mceq])
        
        is_mixed = np.isin(wstack.pid, self.mceq_mixed_pdgs)
        is_mixed_inter = np.full_like(is_mixed, False)
        is_mixed_not_inter = np.full_like(is_mixed, False)
        is_mixed_inter[is_mixed] = wstack.energy[is_mixed] > self.get_mix_energy(wstack.pid[is_mixed])
        is_mixed_not_inter[is_mixed] = np.logical_not(is_mixed_inter[is_mixed])
        
        self.final_stack.append(wstack[is_mixed_inter])
        self.final_decay_stack.append(self.set_decay_at_place(wstack[is_mixed_not_inter]))
        
        is_resonance = np.isin(wstack.pid, self.mceq_resonance_pdgs)
        self.final_decay_stack.append(self.set_decay_at_place(wstack[is_resonance]))
        
        is_inter = np.isin(wstack.pid, self.mceq_only_interacting_pdgs)
        self.final_stack.append(wstack[is_inter])
        
    
    def handle_above_threshold(self):
        wstack = self.above_threshold_stack.valid()
        
        is_mceq = np.isin(wstack.pid, self.pdg_lists.mceq_particles)
        is_not_mceq = np.logical_not(is_mceq)
        
        self.decay_stack.append(wstack[is_not_mceq])
        
        is_resonance = np.isin(wstack.pid, self.mceq_resonance_pdgs)
        self.decay_stack.append(self.set_decay_at_place(wstack[is_resonance]))
        
        
        is_mixed = np.isin(wstack.pid, self.mceq_mixed_pdgs)
        is_mixed_inter = np.full_like(is_mixed, False)
        is_mixed_not_inter = np.full_like(is_mixed, False)
        is_mixed_inter[is_mixed] = wstack.energy[is_mixed] > self.get_mix_energy(wstack.pid[is_mixed])
        is_mixed_not_inter[is_mixed] = np.logical_not(is_mixed_inter[is_mixed])
        
        self.decay_stack.append(self.set_decay_at_place(wstack[is_mixed_not_inter]))
        
        self.propagating_stack.clear()
        self.propagating_stack.append(wstack[is_mixed_inter])
        is_inter = np.isin(wstack.pid, self.mceq_only_interacting_pdgs)
        self.propagating_stack.append(wstack[is_inter])        
    
    
    def filter_by_slant_depth(self):
        
        if len(self.propagating_stack) == 0:
            self.inter_stack = None
            return
        
        pstack = self.propagating_stack.valid()    
        
        self.xdepth_getter.get_decay_xdepth(pstack)
        self.xdepth_getter.get_inter_xdepth(pstack)
        
        max_xdepth = self.stop_xdepth
                
        # Sort particles at the surface
        at_surface = np.logical_and(pstack.xdepth_inter >= max_xdepth, 
                                    pstack.xdepth_decay >= max_xdepth)
        
        not_at_surface = np.logical_not(at_surface)
        
        
        should_not_decay = np.isin(pstack.pid, self.pdg_lists.mceq_particles)
        should_decay = np.logical_not(should_not_decay)
        
        should_decay = np.where(np.logical_and(at_surface, should_decay))[0]
        should_not_decay = np.where(np.logical_and(at_surface, should_not_decay))[0]
        
        
        # Particles that should be decayed at surface
        dfstack_portion = pstack[should_decay]
        dfstack_portion.xdepth_stop[:] = max_xdepth
        dfstack_portion.final_code[:] = 1
        self.final_decay_stack.append(dfstack_portion)
        
        # Particles that are already at their final stage
        fstack_portion = pstack[should_not_decay]
        fstack_portion.xdepth_stop[:] = max_xdepth
        fstack_portion.final_code[:] = 1
        self.final_stack.append(fstack_portion)

        # Sort particles which are still in the atmosphere
        istack_true = np.logical_and(pstack.xdepth_inter < pstack.xdepth_decay, not_at_surface)
        istack_true = np.where(istack_true)[0]
        istack_portion = pstack[istack_true]        
        istack_portion.xdepth_stop[:] = istack_portion.xdepth_inter
        self.inter_stack = istack_portion
        
        dstack_true = np.logical_and(pstack.xdepth_inter > pstack.xdepth_decay, not_at_surface)
        dstack_true = np.where(dstack_true)[0]
        dstack_portion = pstack[dstack_true]
        dstack_portion.xdepth_stop[:] = dstack_portion.xdepth_decay
        self.decay_stack.append(dstack_portion)
    

    def run_hadron_interactions(self):
        self.children_stack.clear()
        self.rejection_stack.clear()
        self.working_stack.clear()
        
        if (self.inter_stack is None) or (len(self.inter_stack) == 0):
            return
        
        self.number_of_interactions += self.hadron_interaction.run_event_generator(
                                                    parents = self.inter_stack, 
                                                    children = self.children_stack, 
                                                    failed_parents = self.rejection_stack)
                
        self.id_generator.generate_ids(self.children_stack.valid().id)
        chstack = self.children_stack.valid()
        chstack.production_code[:] = 2
        
        # Filter particles participated in interactions
        parents_true = np.logical_not(np.isin(self.inter_stack.valid().id, 
                                              self.rejection_stack.valid().id))
        parents = self.inter_stack[np.where(parents_true)[0]]
        parents.valid().final_code[:] = 2
        # And record them in archival stack
        self.archival_stack.append(parents)
        self.working_stack.append(chstack)
                
        
        if len(self.rejection_stack) > 0:
            rstack = self.rejection_stack.valid()
            self.archival_stack.append(rstack)
            self.decay_stack.append(rstack)
        

    
    def run_particle_decay(self):
        if len(self.decay_stack) == 0:
            return
        
        self.number_of_decays += self.decay_driver.run_decay(self.decay_stack,
                                                             decay_products=self.working_stack,
                                                             decayed_particles=self.spare_working_stack,
                                                             stable_particles=self.rejection_stack)
        
        
        # Record decayed particles
        # Final_code = 3 means decay
        self.spare_working_stack.valid().final_code = 3
        # And record them in archival stack
        self.archival_stack.append(self.spare_working_stack)
        
        
        
        self.id_generator.generate_ids(self.working_stack.valid().id)
    
        self.rejection_stack.valid().xdepth_stop[:] = self.stop_xdepth
        self.final_stack.append(self.rejection_stack)         
        self.decay_stack.clear()
        
        
    def run_final_forced_decay(self):
        
        while len(self.final_decay_stack) > 0:            
            self.number_of_decays += self.decay_driver.run_decay(self.final_decay_stack,
                                                                decay_products=self.working_stack, 
                                                                decayed_particles=self.spare_working_stack,
                                                                stable_particles=self.rejection_stack)
        
        
            self.final_decay_stack.clear()
            wstack = self.working_stack.valid()
            self.id_generator.generate_ids(wstack.id)
            
            is_mixed = np.isin(wstack.pid, self.mceq_mixed_pdgs)
            is_mixed_inter = np.full_like(is_mixed, False)
            is_mixed_not_inter = np.full_like(is_mixed, False)
            is_mixed_inter[is_mixed] = wstack.energy[is_mixed] > self.get_mix_energy(wstack.pid[is_mixed])
            is_mixed_not_inter[is_mixed] = np.logical_not(is_mixed_inter[is_mixed])
            
            self.final_stack.append(wstack[is_mixed_inter])
            self.final_decay_stack.append(self.set_decay_at_place(wstack[is_mixed_not_inter]))
            
            is_inter = np.isin(wstack.pid, self.mceq_only_interacting_pdgs)
            self.final_stack.append(wstack[is_inter])
            
            is_resonance = np.isin(wstack.pid, self.mceq_resonance_pdgs)
            self.final_decay_stack.append(self.set_decay_at_place(wstack[is_resonance]))
            
            is_uncond_final = np.isin(wstack.pid, self.unconditionally_final_pdgs)
            self.final_stack.append(wstack[is_uncond_final])
            
            is_not_mceq = np.logical_not(np.isin(wstack.pid, self.pdg_lists.mceq_particles))
            self.final_decay_stack.append(wstack[is_not_mceq])
            self.final_stack.append(self.rejection_stack)
        
    def get_decaying_particles(self):
        return self.decay_stack
    
    def get_final_particles(self):
        return self.final_stack        
    

    

        
        
    
    
    
    