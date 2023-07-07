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





# def example_interaction_model():
#     import chromo
#     target = chromo.kinematics.CompositeTarget([("N", 0.78), ("O", 0.22)])
#     ekin = chromo.kinematics.FixedTarget(1e7, "O16", target)
#     model = chromo.models.DpmjetIII191
    
#     return InteractionModel(model, ekin, target)

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
        self.generated_stack = ParticleArray()
        
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
        
        
        # self.stable_pdgs = unique_pdgs_np(stable_pdgs)
        # self.decay_not_interact_pdgs = unique_pdgs_np(decay_not_interact_pdgs)
        # self.decay_when_final_pdgs = unique_pdgs_np(decay_when_final_pdgs)  
        self.decay_at_vertex_pdgs = unique_sorted_ids([])
        
        is_final_pdgs = np.logical_not(np.isin(self.pdg_lists.mceq_particles, 
                                                                self.mceq_resonance_pdgs))
        self.final_pdgs = unique_sorted_ids(self.pdg_lists.mceq_particles[is_final_pdgs])
        
        
        # self._check_for_duplicates()    
        # By default all particles that can physically decay will decay in pythia
        # pdgs in stable_pdgs will not be decayed in pythia
        self.decay_driver.set_decaying_pdgs(decaying_pdgs=self.phys_decaying_pdgs,
                                            # stable_pdgs=self.stable_pdgs
                                            )  
        
    
    def get_mix_energy(self, pdg):
        return self.mceq_mixed_energy[self.mceq_mixed_map.get_pids(pdg)]
            
        
    # def _check_for_duplicates(self):
    #     """Raise exception if any pdg id is more than in one
    #     mutually exclusive arrays
    #     """
    #     all_pdgs = np.concatenate([self.stable_pdgs, 
    #                    self.decay_not_interact_pdgs, 
    #                    self.decay_when_final_pdgs,
    #                    self.decay_at_vertex_pdgs])
        
    #     values, counts = np.unique(all_pdgs, return_counts=True)
    #     duplicates = values[counts > 1]
    #     if len(duplicates) > 0:
    #         raise ValueError("Duplicates in mutually exclusive arrays for:\n"
    #                           f"pdg_id={duplicates} are met\n"
    #                           f"times={counts[counts > 1]}")
        
    
    
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
        print(f"stop depth = {self.stop_xdepth}")
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
            self.generated_stack.clear()
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
        self.generated_stack.append(self.working_stack)
        
        iloop = 1
        
        
        start_time = time.time()        
        while len(self.working_stack) > 0:
            
            # print(f"\r{iloop} Number of inter = {self.number_of_interactions}"
            #       f" number of decays = {self.number_of_decays}")
            
            # print(f"{iloop} Working stack = {len(self.working_stack)}")
            # self.catch_charged_pions(1)
            self.filter_uncond_final()
            # self.catch_charged_pions(2)
            self.filter_by_threshold_energy()
            # self.catch_charged_pions(3)
            self.handle_below_threshold()
            # self.catch_charged_pions(4)
            self.handle_above_threshold()
            # self.catch_charged_pions(5)
            self.filter_by_slant_depth2()
            # self.catch_charged_pions(6)
            # self.filter_out_final()
            # self.filter_by_slant_depth1()         
            self.run_hadron_interactions()
            # self.catch_charged_pions(7)
            if len(self.working_stack) == 0:
                self.run_particle_decay()
                # self.catch_charged_pions(8)
            
            iloop += 1
        
        self.run_final_forced_decay()
        # self.catch_charged_pions(9)
        
        self.loop_execution_time += time.time() - start_time
        self.runs_number += 1
    
    
    
    def filter_uncond_final(self):
        wstack = self.working_stack.valid()
        is_uncond_final = np.isin(wstack.pid, self.unconditionally_final_pdgs)
        is_not_uncond_final = np.logical_not(is_uncond_final)
        self.final_stack.append(wstack[is_uncond_final])
        
        self.spare_working_stack.clear()
        self.spare_working_stack.append(wstack[is_not_uncond_final])
        self.working_stack.clear()
    
    def filter_by_threshold_energy(self):
        wstack = self.spare_working_stack.valid()
        
        is_above_threshold = wstack.energy > self.threshold_energy
        is_below_threshold = np.logical_not(is_above_threshold)
        
        self.above_threshold_stack.clear()
        self.above_threshold_stack.append(wstack[is_above_threshold])
        
        self.below_threshold_stack.clear()
        self.below_threshold_stack.append(wstack[is_below_threshold])
        
        self.spare_working_stack.clear()
        
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
        self.final_decay_stack.append(wstack[is_mixed_not_inter])
        
        is_resonance = np.isin(wstack.pid, self.mceq_resonance_pdgs)
        self.final_decay_stack.append(wstack[is_resonance])
        
        is_inter = np.isin(wstack.pid, self.mceq_only_interacting_pdgs)
        self.final_stack.append(wstack[is_inter])
        
        self.test_intersecting_sets([is_not_mceq, is_resonance, is_mixed_inter, is_mixed_not_inter, is_inter],
                                    wstack)
        
    
    def handle_above_threshold(self):
        wstack = self.above_threshold_stack.valid()
        
        is_mceq = np.isin(wstack.pid, self.pdg_lists.mceq_particles)
        is_not_mceq = np.logical_not(is_mceq)
        
        self.decay_stack.append(wstack[is_not_mceq])
        
        is_resonance = np.isin(wstack.pid, self.mceq_resonance_pdgs)
        self.decay_stack.append(wstack[is_resonance])
        
        
        is_mixed = np.isin(wstack.pid, self.mceq_mixed_pdgs)
        is_mixed_inter = np.full_like(is_mixed, False)
        is_mixed_not_inter = np.full_like(is_mixed, False)
        is_mixed_inter[is_mixed] = wstack.energy[is_mixed] > self.get_mix_energy(wstack.pid[is_mixed])
        is_mixed_not_inter[is_mixed] = np.logical_not(is_mixed_inter[is_mixed])
        
        self.decay_stack.append(wstack[is_mixed_not_inter])
        
        self.propagating_stack.clear()
        self.propagating_stack.append(wstack[is_mixed_inter])
        is_inter = np.isin(wstack.pid, self.mceq_only_interacting_pdgs)
        self.propagating_stack.append(wstack[is_inter])
        
         
        self.test_intersecting_sets([is_not_mceq, is_resonance, is_mixed_inter, is_mixed_not_inter, is_inter],
                                    wstack)
        
    def test_intersecting_sets(self, bool_arrays, wstack):
        res = None
        
        for arr in bool_arrays:
            if res is None:
                res = arr.astype(int)
            else:
                res += arr.astype(int)
         
        if np.any(res != 1):
            print(f"Intersection of arrays = {res[res != 1]}")
            print(f"Pid = {wstack.pid[res != 1]}")
            print(f"Energy = {wstack.energy[res != 1]}")               
    
    
    def catch_charged_pions(self, place):
        if self.cought_pions is None:
                self.cought_pions = 0
        
        wstack = self.final_stack.valid()  
        
        charged_pions = np.isin(wstack.pid, np.array([111]))
        
        below_mixed = np.where(wstack.energy[charged_pions] <= self.get_mix_energy(wstack.pid[charged_pions]))[0]
        
        if len(below_mixed) != self.cought_pions:
            print(f"{place} Charged pions below mixed en = {len(below_mixed)}")
            self.cought_pions = len(below_mixed)
            
    
    def filter_out_final(self):
        
        wstack = self.working_stack.valid()
        
        # Filter unconditionally final
        is_uncond_final = np.isin(wstack.pid, self.unconditionally_final_pdgs)
        self.final_stack.append(wstack[is_uncond_final])
        is_not_uncond_final = np.logical_not(is_uncond_final)
        
        # Filter by energy threshold
        is_above_threshold = wstack.energy > self.threshold_energy
        is_below_threshold = np.logical_not(is_above_threshold)
        
        is_above_threshold = np.logical_and(is_not_uncond_final, is_above_threshold)
        is_below_threshold = np.logical_and(is_not_uncond_final, is_below_threshold)
        
    
        
        is_mixed = np.isin(wstack.pid, self.mceq_mixed_pdgs)
        is_mixed_inter = np.full_like(is_mixed, False)
        is_mixed_not_inter = np.full_like(is_mixed, False)
        is_mixed_inter[is_mixed] = wstack.energy[is_mixed] > self.get_mix_energy(wstack.pid[is_mixed])
        is_mixed_not_inter[is_mixed] = np.logical_not(is_mixed_inter[is_mixed])
        # is_mixed = np.isin(pstack.pid, self.mceq_mixed_pdgs)
        
        is_mixed_inter_below = np.logical_not(is_below_threshold, is_mixed_inter)
        is_mixed_inter_above = np.logical_not(is_above_threshold, is_mixed_inter)
        
        is_mixed_not_inter_below = np.logical_not(is_below_threshold, is_mixed_not_inter)
        is_mixed_not_inter_above = np.logical_not(is_above_threshold, is_mixed_not_inter)
        
        self.final_stack.append(wstack[is_mixed_inter_below])
        self.final_decay_stack.append(wstack[is_mixed_not_inter_below])
        
        self.decay_stack.append(wstack[is_mixed_not_inter_above])
        self.propagating_stack.clear()
        self.propagating_stack.append(wstack[is_mixed_inter_above])
        
        
        # Split particles below threshold to final and other (assume that they are decaying)
        is_final = np.isin(wstack.pid, self.final_pdgs)
        is_final_not_mixed = np.logical_and(is_final, np.logical_not(is_mixed))
        # is_not_final_not_mixed = 
        
        is_not_final = np.logical_not(is_final)
        is_final = np.logical_and(is_below_threshold, is_final)
        is_not_final = np.logical_and(is_below_threshold, is_not_final)
        
        self.final_stack.append(wstack[is_final])
        self.final_decay_stack.append(wstack[is_not_final])
        
        # Split particles above threshold to interacting and decaying and other
        # Other are assumed to only decaying. Decay point is different from production point 
        is_interacting_decaying = np.isin(wstack.pid, self.interacting_decaying_pdgs)
        is_not_interacting_decaying = np.logical_not(is_interacting_decaying)
        is_interacting_decaying = np.logical_and(is_above_threshold, is_interacting_decaying)
        is_not_interacting_decaying = np.logical_and(is_above_threshold, is_not_interacting_decaying)
        
        self.propagating_stack.clear()
        self.propagating_stack.append(wstack[is_interacting_decaying])
        self.decay_stack.append(wstack[is_not_interacting_decaying])
        
        self.working_stack.clear()
    
    
    def filter_by_slant_depth2(self):
        
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
        
        self.working_stack.clear()
    
    def filter_by_slant_depth1(self):
        if len(self.propagating_stack) == 0:
            self.inter_stack = None
            return
        
        pstack = self.propagating_stack.valid()
        
        is_mixed = np.isin(pstack.pid, self.mceq_mixed_pdgs)
        is_inter0 = np.isin(pstack.pid, self.mceq_only_interacting_pdgs)
        
        is_mixed_inter = np.full_like(is_mixed, False)
        is_mixed_inter[is_mixed] = pstack.energy[is_mixed] > self.get_mix_energy(pstack.pid[is_mixed])
        
        is_inter = np.logical_or(is_mixed_inter, is_inter0)
        inter_stack = pstack[is_inter]
        self.decay_stack.append(pstack[np.logical_not(is_inter)])
        
        self.xdepth_getter.get_decay_xdepth(inter_stack)
        self.xdepth_getter.get_inter_xdepth(inter_stack)
        
        max_xdepth = self.stop_xdepth
                
        # Sort particles at the surface
        at_surface = np.logical_and(inter_stack.xdepth_inter >= max_xdepth, 
                                    inter_stack.xdepth_decay >= max_xdepth)
        
        not_at_surface = np.logical_not(at_surface)
        
        should_not_decay = np.isin(inter_stack.pid, self.pdg_lists.mceq_particles)
        should_decay = np.logical_not(should_not_decay)
        
        should_decay = np.where(np.logical_and(at_surface, should_decay))[0]
        should_not_decay = np.where(np.logical_and(at_surface, should_not_decay))[0]
        
        # Particles that should be decayed at surface
        dfstack_portion = inter_stack[should_decay]
        dfstack_portion.xdepth_stop[:] = max_xdepth
        dfstack_portion.final_code[:] = 1
        self.final_decay_stack.append(dfstack_portion)
        
        # Particles that are already at their final stage
        fstack_portion = inter_stack[should_not_decay]
        fstack_portion.xdepth_stop[:] = max_xdepth
        fstack_portion.final_code[:] = 1
        self.final_stack.append(fstack_portion)
        
        # Sort particles which are still in the atmosphere
        istack_true = np.logical_and(inter_stack.xdepth_inter < inter_stack.xdepth_decay, not_at_surface)
        istack_true = np.where(istack_true)[0]
        istack_portion = inter_stack[istack_true]        
        istack_portion.xdepth_stop[:] = istack_portion.xdepth_inter
        self.inter_stack = istack_portion
        
        dstack_true = np.logical_and(inter_stack.xdepth_inter > inter_stack.xdepth_decay, not_at_surface)
        dstack_true = np.where(dstack_true)[0]
        dstack_portion = inter_stack[dstack_true]
        dstack_portion.xdepth_stop[:] = dstack_portion.xdepth_decay
        self.decay_stack.append(dstack_portion)
        
        self.working_stack.clear()
        
         
    
    
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
        
        self.working_stack.clear()

        
        
        
    
    def run_hadron_interactions(self):
        self.children_stack.clear()
        self.rejection_stack.clear()
        
        # print(f"Interaction stack:\npid = {self.inter_stack.valid().pid}")
        # print(f"id = {self.inter_stack.valid().id}")
        # input()
        
        if (self.inter_stack is None) or (len(self.inter_stack) == 0):
            return
        
        self.number_of_interactions += self.hadron_interaction.run_event_generator(
                                                    parents = self.inter_stack, 
                                                    children = self.children_stack, 
                                                    failed_parents = self.rejection_stack)
        
        # Take only MCEq particles
        # valid_children = self.children_stack.valid()
        # is_mceq_particles = np.isin(valid_children.pid, self.pdg_lists.mceq_particles)
        # self.children_stack.clear()
        # self.children_stack.append(valid_children[is_mceq_particles])
        
        self.id_generator.generate_ids(self.children_stack.valid().id)
        chstack = self.children_stack.valid()
        chstack.production_code[:] = 2
        
        self.generated_stack.append(self.children_stack)
        
        # Filter particles participated in interactions
        parents_true = np.logical_not(np.isin(self.inter_stack.valid().id, 
                                              self.rejection_stack.valid().id))
        parents = self.inter_stack[np.where(parents_true)[0]]
        parents.valid().final_code[:] = 2
        # And record them in archival stack
        self.archival_stack.append(parents)
        
        self.working_stack.clear()
        if len(self.decay_at_vertex_pdgs) > 0:
            is_decay_at_vertex = np.isin(chstack.pid, self.decay_at_vertex_pdgs)
            chstack.filter_code[is_decay_at_vertex] = FilterCode.XD_DECAY_ON.value
            chstack.xdepth_decay[is_decay_at_vertex] = chstack.xdepth[is_decay_at_vertex]
            self.decay_stack.append(chstack[is_decay_at_vertex])
               
            self.working_stack.append(chstack[np.logical_not(is_decay_at_vertex)])
        else:
            self.working_stack.append(chstack)
                
        
        if len(self.rejection_stack) > 0:
            rstack = self.rejection_stack.valid()
            self.archival_stack.append(rstack)
            
            # is_stable_lepton = np.isin(rstack.pid, self.pdg_lists.leptons_stable_mceq)
            # is_decaying_lepton = np.isin(rstack.pid, self.pdg_lists.leptons_decay_mceq)
            # is_mixing_hadron = np.isin(rstack.pid, self.pdg_lists.hadrons_mix_mceq)
            # is_stable_hadron = np.isin(rstack.pid, self.pdg_lists.hadrons_stable_mceq)
        
        
            # self.final_stack.append(rstack[is_stable_lepton])
            # self.final_stack.append(rstack[is_mixing_hadron])
            # self.final_stack.append(rstack[is_stable_hadron])
            # self.decay_stack.append(rstack[is_decaying_lepton])
            
            
            self.decay_stack.append(rstack)
            
            rej_muons = np.where(np.isin(self.rejection_stack.valid().pid, 
                                         np.array([-13, 13], dtype = np.int32)))[0]
            if len(rej_muons) > 0:
                print(f"Muons pdgs = {self.rejection_stack.pid[rej_muons]}"
                      f" energy = {self.rejection_stack.energy[rej_muons]}"
                      f" xdepth = {self.rejection_stack.xdepth[rej_muons]}"
                      f" xdepth_decay = {self.rejection_stack.xdepth_decay[rej_muons]}"
                      f" xdepth_inter = {self.rejection_stack.xdepth_inter[rej_muons]}")
        

    
    def run_particle_decay(self):
        if len(self.decay_stack) == 0:
            return
        
        self.rejection_stack.clear()
        self.working_stack.clear()
        self.number_of_decays += self.decay_driver.run_decay(self.decay_stack, 
                                                             decayed_particles=self.working_stack,
                                                             stable_particles=self.rejection_stack)
        
        
        # Filter particles participated in decay
        parents_true = np.logical_not(np.isin(self.decay_stack.valid().id, 
                                              self.rejection_stack.valid().id))
        parents = self.decay_stack[np.where(parents_true)[0]]
        # Final_code = 3 means decay
        parents.valid().final_code[:] = 3
        # And record them in archival stack
        self.archival_stack.append(parents)
        
        self.id_generator.generate_ids(self.working_stack.valid().id)
        self.generated_stack.append(self.working_stack)
        
        self.rejection_stack.valid().xdepth_stop[:] = self.stop_xdepth
        
        
        # self.debug_append_final_stack(self.rejection_stack)
        # print(f"Append rejection_stack to final_stack = {len(self.rejection_stack)}")
        self.final_stack.append(self.rejection_stack) 
        
        # Use rejection_stack again
        self.rejection_stack.clear()
        should_be_finals = np.isin(self.working_stack.valid().final_code, np.array([1, 4], dtype=np.int32))
        should_be_processed = np.logical_not(should_be_finals)
        self.final_stack.append(self.working_stack[np.where(should_be_finals)[0]])
        self.rejection_stack.append(self.working_stack[np.where(should_be_processed)[0]])
        self.working_stack.clear()
        
        self.working_stack.append(self.rejection_stack)  
        
        self.decay_stack.clear()
        
        
    def run_final_forced_decay(self):
        
        while len(self.final_decay_stack) > 0:

            self.rejection_stack.clear()
            self.working_stack.clear()
            
            self.number_of_decays += self.decay_driver.run_decay(self.final_decay_stack, 
                                                                decayed_particles=self.working_stack,
                                                                stable_particles=self.rejection_stack)
        
        
            wstack = self.working_stack.valid()
            self.id_generator.generate_ids(wstack.id)
            
            self.final_decay_stack.clear()
            is_mixed = np.isin(wstack.pid, self.mceq_mixed_pdgs)
            is_mixed_inter = np.full_like(is_mixed, False)
            is_mixed_not_inter = np.full_like(is_mixed, False)
            is_mixed_inter[is_mixed] = wstack.energy[is_mixed] > self.get_mix_energy(wstack.pid[is_mixed])
            is_mixed_not_inter[is_mixed] = np.logical_not(is_mixed_inter[is_mixed])
            
            self.final_stack.append(wstack[is_mixed_inter])
            self.final_decay_stack.append(wstack[is_mixed_not_inter])
            
            is_inter = np.isin(wstack.pid, self.mceq_only_interacting_pdgs)
            self.final_stack.append(wstack[is_inter])
            
            is_resonance = np.isin(wstack.pid, self.mceq_resonance_pdgs)
            self.final_decay_stack.append(wstack[is_resonance])
            
            is_uncond_final = np.isin(wstack.pid, self.unconditionally_final_pdgs)
            self.final_stack.append(wstack[is_uncond_final])
            
            is_not_mceq = np.logical_not(np.isin(wstack.pid, self.pdg_lists.mceq_particles))
            self.final_decay_stack.append(wstack[is_not_mceq])
            self.final_stack.append(self.rejection_stack)
        
    def get_decaying_particles(self):
        return self.decay_stack
    
    def get_final_particles(self):
        return self.final_stack        
    

    

        
        
    
    
    
    