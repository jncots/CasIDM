import numpy as np
import contextlib

#import solver related modules
from MCEq.core import MCEqRun
import mceq_config as config
#import primary model choices
import crflux.models as pm
from casidm.mceq_utils.mceq_solve_rhs import solve_rhs
from casidm.mceq_utils.grid_collector import MceqGridCollector
from casidm.cascade.cascade_driver import CascadeDriver



class MCEQDistributions():
    def __init__(self,
                 energy,
                 pdg_id,
                 theta_deg,
                 slant_depths,
                 energy_range,
                 pname_tuples,
                 interaction_model = "DPMJET-III-19.1", 
                 generic_losses_all_charged = True,
                 enable_energy_loss = True, 
                 muon_helicity_dependence = True,
                 disable_decays = [],
                 hybrid_crossover = 0.5,
                 density_model = ("CORSIKA", ("USStd", None))):
        
        config.mceq_db_fname = "mceq_db_lext_dpm191_v150.h5"
        config.generic_losses_all_charged = generic_losses_all_charged
        config.enable_energy_loss = enable_energy_loss
        config.muon_helicity_dependence = muon_helicity_dependence
        config.adv_set["disable_decays"] = disable_decays
        config.hybrid_crossover = hybrid_crossover
        
        config.e_min = energy_range[0]
        config.e_max = energy_range[1]
        
        # config.enable_2D = True
        # config.mceq_db_fname = 'mceq_db_rare_decays_URQMD_lext_2D.h5'
        # config.mceq_db_fname = "mceq_db_lext_dpm191_v150.h5"
        # config.enable_default_tracking = False
        # config.enable_em = False
        # config.enable_em_ion = False
        # config.hybrid_crossover = 0.1
        # config.muon_energy_loss = True
        # config.enable_cont_rad_loss = True
        # config.enable_energy_loss = True
        # config.muon_helicity_dependence = True
        # config.density_model = ("CORSIKA", ("USStd", None))

        # config.adv_set['force_resonance'] = [421, 431, 411, 310]
        # config.adv_set['disabled_particles'] = [22, 111, 16, 11]
        
        # print(json.dumps(config.adv_set, indent = 4))
        
        mceq_run = MCEqRun(
            #provide the string of the interaction model
            interaction_model=interaction_model,
            #primary cosmic ray flux model
            primary_model = (pm.HillasGaisser2012, "H3a"),
            # Zenith angle in degrees. 0=vertical, 90=horizontal
            theta_deg=theta_deg,
            density_model = density_model
        )

        #obtain energy grid (fixed) of the solution for the x-axis of the plots
        self.e_grid = mceq_run.e_grid
        self.e_widths = mceq_run.e_widths
        self.e_bins = mceq_run.e_bins
        
        self.slant_depths = np.array(slant_depths)

        if mceq_run.density_model.max_X < np.max(self.slant_depths):
            raise ValueError(f"Maximum slant_xdepth = {mceq_run.density_model.max_X}")

        #Set the zenith angle
        mceq_run.set_theta_deg(theta_deg)
        mceq_run.set_single_primary_particle(energy, pdg_id = pdg_id)
        mceq_run.solve(int_grid=self.slant_depths)

        self.mceq_run = mceq_run

        # Populate longitudinal spectra for all particles:
        part_long_spectra = {}
        for p in mceq_run.pman.all_particles:
            # print(f"Spectrum for {p.name}")
            part_long_xdepth = {}
            try:
                for ixdepth in range(len(self.slant_depths)):
                    part_long_xdepth[ixdepth] = mceq_run.get_solution(p.name, grid_idx=ixdepth)
                part_long_spectra[p.name] = part_long_xdepth    
            except Exception as ex:
                pass
                # print(ex)    

        # print(f"part_long_spectra.keys = {[k for k in part_long_spectra.keys()]}")
        
        # all_category_names = [k for k in part_long_spectra.keys()]
        # for cat_name in all_category_names:
        #     if "_mu" in cat_name:
        #         print(cat_name)


        self.flux_depth = dict()
        for ixdepth in range(len(self.slant_depths)):
            
            self.flux = dict()
            for pnames in pname_tuples:
                
                group_name = pnames[0]
                self.flux[group_name] = None
                for pname in pnames[1:]:
                    if self.flux[group_name] is None:
                        try:
                            self.flux[group_name] = part_long_spectra[pname][ixdepth]
                        except:
                            self.flux[group_name] = 0  
                    else:
                        try:
                            self.flux[group_name] += part_long_spectra[pname][ixdepth]
                        except:
                            self.flux[group_name] += 0    
                
                self.flux[group_name] = self.flux[group_name] * self.e_widths

            self.flux_depth[ixdepth] = self.flux
        
        self.flux = self.flux_depth
        
        
class MCEQExtractDists():
    def __init__(self, mceq_run, 
                 pname_tuples,
                 slant_depths):
        self.mceq_run = mceq_run
        
        #obtain energy grid (fixed) of the solution for the x-axis of the plots
        self.e_grid = mceq_run.e_grid
        self.e_widths = mceq_run.e_widths
        self.e_bins = mceq_run.e_bins
        
        self.slant_depths = np.array(slant_depths)

        if mceq_run.density_model.max_X < np.max(self.slant_depths):
            raise ValueError(f"Maximum slant_xdepth = {mceq_run.density_model.max_X}")

        # Populate longitudinal spectra for all particles:
        part_long_spectra = {}
        for p in mceq_run.pman.all_particles:
            # print(f"Spectrum for {p.name}")
            part_long_xdepth = {}
            try:
                for ixdepth in range(len(self.slant_depths)):
                    part_long_xdepth[ixdepth] = mceq_run.get_solution(p.name, grid_idx=ixdepth)
                part_long_spectra[p.name] = part_long_xdepth    
            except Exception as ex:
                pass
                # print(ex)    

        # print(f"part_long_spectra.keys = {[k for k in part_long_spectra.keys()]}")
        
        # all_category_names = [k for k in part_long_spectra.keys()]
        # for cat_name in all_category_names:
        #     if "_mu" in cat_name:
        #         print(cat_name)


        self.flux_depth = dict()
        for ixdepth in range(len(self.slant_depths)):
            
            self.flux = dict()
            for pnames in pname_tuples:
                
                group_name = pnames[0]
                self.flux[group_name] = None
                for pname in pnames[1:]:
                    if self.flux[group_name] is None:
                        try:
                            self.flux[group_name] = part_long_spectra[pname][ixdepth]
                        except:
                            self.flux[group_name] = 0  
                    else:
                        try:
                            self.flux[group_name] += part_long_spectra[pname][ixdepth]
                        except:
                            self.flux[group_name] += 0    
                
                self.flux[group_name] = self.flux[group_name] * self.e_widths

            self.flux_depth[ixdepth] = self.flux
        
        self.flux = self.flux_depth
        
        
class MceqWrapper():
    def __init__(self,
               pdg_id,
               energy,
               theta_deg,
               energy_range,
               slant_depths,
               density_model = ("CORSIKA", ("USStd", None)), 
               interaction_model = "DPMJET-III-19.1",
               generic_losses_all_charged = True,
               enable_energy_loss = True, 
               muon_helicity_dependence = True,
               hybrid_crossover = 0.5,
               disable_decays = []):
        
        self.pdg_id = pdg_id
        self.energy = energy
        self.theta_deg = theta_deg
        
        self.e_min = energy_range[0]
        self.e_max = energy_range[1]
        self.generic_losses_all_charged = generic_losses_all_charged
        self.enable_energy_loss = enable_energy_loss
        self.muon_helicity_dependence = muon_helicity_dependence
        self.disable_decays = disable_decays
        self.hybrid_crossover = hybrid_crossover
        
        self._set_config()

        
        with contextlib.redirect_stdout(None):
            self.mceq_run = MCEqRun(
                #provide the string of the interaction model
                interaction_model=interaction_model,
                #primary cosmic ray flux model
                primary_model = (pm.HillasGaisser2012, "H3a"),
                # Zenith angle in degrees. 0=vertical, 90=horizontal
                theta_deg=self.theta_deg,
                density_model = density_model
            )
        
        
        
        #obtain energy grid (fixed) of the solution for the x-axis of the plots
        self.e_grid = self.mceq_run.e_grid
        self.e_widths = self.mceq_run.e_widths
        self.e_bins = self.mceq_run.e_bins
        self._set_slant_depths(slant_depths)
        self._particle_lists()
        self.collector = MceqGridCollector(self)
        
   
    def _save_config(self):
        old_config = {}
        old_config["e_min"] = config.e_min
        old_config["e_max"] = config.e_max
        old_config["generic_losses_all_charged"] = config.generic_losses_all_charged
        old_config["enable_energy_loss"] = config.enable_energy_loss
        old_config["muon_helicity_dependence"] = config.muon_helicity_dependence
        old_config["disable_decays"] = config.adv_set["disable_decays"]
        old_config["hybrid_crossover"] = config.hybrid_crossover
        self.old_config = old_config
    
    def restore_config(self):
        old_config = self.old_config
        config.e_min = old_config["e_min"]
        config.e_max = old_config["e_max"] 
        
        config.generic_losses_all_charged = old_config["generic_losses_all_charged"]
        config.enable_energy_loss = old_config["enable_energy_loss"]
        config.muon_helicity_dependence = old_config["muon_helicity_dependence"]
        config.adv_set["disable_decays"] = old_config["disable_decays"]
        config.hybrid_crossover = old_config["hybrid_crossover"]
            
   
    def _set_config(self):
        
        self._save_config()
        
        config.e_min = self.e_min
        config.e_max = self.e_max
        
        # Advanced settings
        config.generic_losses_all_charged = self.generic_losses_all_charged
        config.enable_energy_loss = self.enable_energy_loss
        config.muon_helicity_dependence = self.muon_helicity_dependence
        config.adv_set["disable_decays"] = self.disable_decays
        config.hybrid_crossover = self.hybrid_crossover
        
        # Additional settings
        
        # config.enable_2D = True
        # config.mceq_db_fname = 'mceq_db_rare_decays_URQMD_lext_2D.h5'
        # config.mceq_db_fname = "mceq_db_lext_dpm191_v150.h5"
        # config.enable_default_tracking = False
        # config.enable_em = False
        # config.enable_em_ion = False
        # config.hybrid_crossover = 0.1
        # config.muon_energy_loss = True
        # config.enable_cont_rad_loss = True
        # config.enable_energy_loss = True
        # config.muon_helicity_dependence = True
        # config.density_model = ("CORSIKA", ("USStd", None))
        # config.adv_set['force_resonance'] = [421, 431, 411, 310]
        # config.adv_set['disabled_particles'] = [22, 111, 16, 11]
        
        
    def _particle_lists(self):        
        collective_pdgs = 1000000
        
        self.pdg_mceqidx_map = {}
        
        final_pdgs = []
        only_interacting_pdgs = []
        only_decaying_pdgs = []
        resonance_pdgs = []
        mixed_pdgs = []
        mixed_tot_energy = []
        
        self.mceq_particles = []   
        for p in self.mceq_run.pman.all_particles:
            pdg_id = p.unique_pdg_id[0]
            chirality = p.unique_pdg_id[1]
            if (abs(pdg_id) < collective_pdgs) and (chirality == 0):
                self.mceq_particles.append((p.name, pdg_id, p.mceqidx, p.E_mix, p.E_crit, p.is_mixed, p.is_resonance, p.mass))
                if p.is_mixed:
                    mixed_pdgs.append(pdg_id)
                    mixed_tot_energy.append(p.E_mix + p.mass)
                elif p.is_resonance:
                    resonance_pdgs.append(pdg_id)
                elif pdg_id in (-2212, 2212):
                    only_interacting_pdgs.append(pdg_id)
                elif pdg_id in (-13, 13):
                    only_decaying_pdgs.append(pdg_id)
                else:
                    final_pdgs.append(pdg_id)
                    
                if p.mceqidx != -1:
                    self.pdg_mceqidx_map[pdg_id] = p.mceqidx
        
        mceq_mixed_info = {}
        mceq_mixed_info["pdg_ids"] = np.array(mixed_pdgs)
        mceq_mixed_info["etot_mix"] = np.array(mixed_tot_energy)
        
        pdgs_categories = {}
        pdgs_categories["final"] = final_pdgs
        pdgs_categories["only_interacting"] = only_interacting_pdgs
        pdgs_categories["only_decaying"] = only_decaying_pdgs
        pdgs_categories["resonance"] = resonance_pdgs
        pdgs_categories["mixed"] = mceq_mixed_info
        self.pdgs_categories = pdgs_categories      
                                  
     
    def _set_slant_depths(self, slant_depths):
        self.slant_depths = np.array(slant_depths)
        if self.mceq_run.density_model.max_X < np.max(self.slant_depths):
            raise ValueError(f"Maximum slant_xdepth = {self.mceq_run.density_model.max_X}")
                                            
        
    def solve_single_particle(self):
        self.mceq_run.set_single_primary_particle(self.energy, pdg_id = self.pdg_id)
        self.mceq_run.solve(int_grid=self.slant_depths)
    
    def solve_collector(self):
        solve_rhs(self.mceq_run, int_grid=self.slant_depths, 
                     grid_var = "X",
                     rhs_source = self.collector.state_vectors())
    
        
    def get_fluxes_for(self, pname_tuples):
        # Populate longitudinal spectra for all particles:
        part_long_spectra = {}
        for p in self.mceq_run.pman.all_particles:
            # print(f"Spectrum for {p.name}")
            part_long_xdepth = {}
            try:
                for ixdepth in range(len(self.slant_depths)):
                    part_long_xdepth[ixdepth] = self.mceq_run.get_solution(p.name, grid_idx=ixdepth)
                part_long_spectra[p.name] = part_long_xdepth    
            except Exception as ex:
                # print(f"{p.name} is not given")
                pass

        self.flux_depth = dict()
        for ixdepth in range(len(self.slant_depths)):
            
            self.flux = dict()
            for pnames in pname_tuples:
                
                group_name = pnames[0]
                self.flux[group_name] = None
                for pname in pnames[1:]:
                    if self.flux[group_name] is None:
                        try:
                            self.flux[group_name] = part_long_spectra[pname][ixdepth]
                        except:
                            self.flux[group_name] = 0  
                    else:
                        try:
                            self.flux[group_name] += part_long_spectra[pname][ixdepth]
                        except:
                            self.flux[group_name] += 0    
                
                self.flux[group_name] = self.flux[group_name] * self.e_widths

            self.flux_depth[ixdepth] = self.flux
        
        self.flux = self.flux_depth
        return self.flux                      
        
        
        
        
class HybridMCEq(MceqWrapper):
    
    def start_cascade_driver(self, interaction_model, threshold):
        self.threshold = threshold
        
        if threshold >= self.energy:
            raise ValueError(f"threshold energy = {threshold:.3e} GeV >= initial energy {self.energy:.3e} GeV ")
        
        self.cascade_driver = CascadeDriver(interaction_model)
        self._restart_cascade_driver()
    
    
        return self.cascade_driver
    
    def _restart_cascade_driver(self):
        self.cascade_driver.simulation_parameters(
                                 pdg = self.pdg_id, energy = self.energy, 
                                 zenith_angle = self.theta_deg, xdepth = 0,
                                 threshold_energy = self.threshold, stop_height = 0,
                                 accumulate_runs = True, reset_ids = True,
                                 pdgs_categories = self.pdgs_categories
                                 )
          
    
    def run_hybrid_solution(self, cascade_runs = 1, restart=False):
        
        if restart:
            self._restart_cascade_driver()
            
        self.cascade_driver.run(cascade_runs)
        
        self.collector.clear()
        self.collector.add_particles(self.cascade_driver.final_stack.valid(),
                                     self.cascade_driver.runs_number)
        self.solve_collector()
    
    def set_result_categories(self, result_categories):
        self.result_categories = result_categories
    
    def hybrid_solution(self, cascade_runs, restart = False):
        self.run_hybrid_solution(cascade_runs, restart)
        self.hybrid_flux = self.get_fluxes_for(self.result_categories)
        return self.hybrid_flux
    
    def regular_solution(self):
        self.solve_single_particle()
        self.regular_flux = self.get_fluxes_for(self.result_categories)
        return self.regular_flux