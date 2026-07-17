"""
Typical usage
-------------
  optimized_dict = optimize(input_dict)
  Optimizes brodge parameters in the input dictionary
"""

from __future__ import annotations

import math

import numpy as np

from osdagbridge.core.bridge_types.plate_girder.plategirderbridge import (
    PlateGirderBridge,
    resolve_girder_value,
)
from osdagbridge.core.bridge_types.plate_girder.analysis_results import (
    PlateGirderAnalysisResults
)
from osdagbridge.core.utils.common import (
    KEY_PROJECT_LOCATION,
    KEY_SPAN,
    KEY_CARRIAGEWAY_WIDTH,
    KEY_DECK_CONCRETE_GRADE_BASIC,
    KEY_TS_NO_OF_GIRDERS,
    KEY_TS_GIRDER_SPACING,
    KEY_TS_DECK_OVERHANG,
    KEY_TS_OVERALL_WIDTH,
    KEY_TS_DECK_THICKNESS,
    KEY_DS_REINF_MATERIAL,
    KEY_MP_GIRDER_SYMMETRY,
    KEY_MP_GIRDER_DEPTH, 
    KEY_MP_GIRDER_TOP_FLANGE_WIDTH, 
    KEY_MP_GIRDER_BOTTOM_FLANGE_WIDTH, 
    KEY_MP_GIRDER_TOP_FLANGE_THICKNESS, 
    KEY_MP_GIRDER_BOTTOM_FLANGE_THICKNESS, 
    KEY_MP_GIRDER_WEB_THICKNESS,
    KEY_MATERIAL_GIRDER_FY,
    KEY_LL_IRC_CLASS_A,
    KEY_LL_IRC_70R_TRACKED,
    KEY_DS_STUD_HEIGHT,
)
# from osdagbridge.core.bridge_types.plate_girder.defaults import BASIC_INPUT_DICT

from osdagbridge.core.bridge_types.plate_girder.designer import (
    SteelSection,
    run_design_check
)

from . import deckdesign
from .defaults import solve_extend_basic_input_dict
# from osdagbridge.core.optimizer.optimizer import Optimizer


# ------------------------------------------------------------------------------
#  Constants
# ------------------------------------------------------------------------------


# IS 2062 standard plate thickness list (mm) — tf and tw must come from here
_STD_PLATES = np.array([
    6, 8, 10, 12, 14, 16, 18, 20, 22, 25,
    28, 32, 36, 40, 45, 50, 56, 63, 70, 80, 90, 100,
])

# ------------------------------------------------------------------------------
#  Utility helpers
# ------------------------------------------------------------------------------


def sqrt(x):
    try:
        return math.sqrt(x)
    except ValueError:
        return 0.0  # Returns Not-a-Number for negative values


def _ceiling_plate(value_mm: float) -> float:
    """Smallest IS 2062 plate thickness ≥ value_mm (always structurally safe)."""
    for p in _STD_PLATES:
        if p >= value_mm:
            return float(p)
    return float(_STD_PLATES[-1])


def _ceil(v: float, n: float = 10.0) -> float:
    """Ceil to nearest multiple of n."""
    return float(math.ceil(v / n) * n)


def _floor(v: float, n: float = 10.0) -> float:
    """Round to nearest multiple of n."""
    return float(math.floor(v / n) * n)


def clamp(v: float, lo: float, hi: float) -> float:
    """Hard-clip v to [lo, hi]."""
    return max(lo, min(v, hi))


# ------------------------------------------------------------------------------
#  NullWriter / mute — silence PlateGirderBridge console output during DE
# ------------------------------------------------------------------------------

class NullWriter:
    """Dummy writer that discards all output."""
    def write(self, text): pass
    def flush(self): pass

"""
def mute(func):
    # Decorator: redirect stdout to NullWriter for the duration of the call.
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = NullWriter()
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
    return wrapper
"""

# TrialPlateGirderBridge which creates PlateGirderBridges with given design vectors

class TrialPlateGirderBridge(PlateGirderBridge):
    
    def __init__(self, input_dict: dict, x: np.ndarray) -> None:
        
        super().__init__()
        solve_extend_basic_input_dict(input_dict, x[0], False)
        
        self.input_dict = dict(input_dict)
        print("current_girder_value: ",self.input_dict[KEY_TS_NO_OF_GIRDERS])

        self.utility_ratio = 0.0
        
        update_dict(self.input_dict, x)
        
        self.set_input(self.input_dict)
            
    
    # @mute
    def design_is_feasible(self) -> str:
        """
        Run the full Osdag grillage + IRC 22:2015 DCR pipeline.
        """
        try:
            # Pre-stage: Unit conversions (must run before validation)
            self._resolve_optimized_bounds_to_mm()
            self._convert_girder_dims_mm_to_m()
            
            # Stage 1: Input Validation
            # self._run_stage("1", self._validate_inputs)
            # self._validate_inputs() # optimiser already validates inputs
            
            # Stage 2: Bridge Layout Solving
            # self._run_stage("2", self._solve_bridge_layout)
            # self._solve_bridge_layout()
            
            # Stage 3: Grillage Setup
            # self._run_stage("3", self._stage_grillage_setup)
            self._stage_grillage_setup()

            # Stage 4A: Dead Load Application
            # self._run_stage("4A", self.add_dead_loads)
            self.add_dead_loads()
            
            # Stage 4B: Live Load Application
            # self._run_stage("4B", self.add_live_loads)
            self.add_live_loads()
            
            # Stage 4C: Wind Load Application
            # self._run_stage("4C", self.add_wind_loads)
            self.add_wind_loads()
            
            # Stage 4D: Temperature Load Application
            # self._run_stage("4D", self.add_temperature_load)
            self.add_temperature_load()
            
            # Stage 4E: Seismic Load Application
            # self._run_stage("4E", self.add_seismic_loads)
            self.add_seismic_loads()
            
            # Stage 4F: Load Combination Envelope
            # self._run_stage("4F", self._stage_load_combinations)
            self._stage_load_combinations()
            
            # Stage 4G: Structural Analysis
            dataset = self._reanalyze_with_dedup()
            dataset = self.create_envelope_load_case(dataset)
            
            edge_dist = self.input_dict[KEY_TS_DECK_OVERHANG]
            results = PlateGirderAnalysisResults(dataset=dataset, bridge=self.grillage_model,edge_dist = edge_dist)
            _, engine, design_results = run_design_check(plate_girder_bridge=self, analysis_results=results, print_report=False)
            
            if engine.overall_status() == "FAIL":
                return "GIRDER FAIL"
            # self.result_data = self.grillage_model.get_result_data()
            
            # Deck Slab design
            concrete_grade = str(self.input_dict[KEY_DECK_CONCRETE_GRADE_BASIC]).strip()
            rebar_grade = str(self.input_dict[KEY_DS_REINF_MATERIAL]).strip()

            fck = self._lookup_material(concrete_grade, "fck")
            Ecm = self._lookup_material(concrete_grade, "Ecm")
            fctm = self._lookup_material(concrete_grade, "fctm")
            fy = self._lookup_material(rebar_grade, "fy")
            Es = self._lookup_material(rebar_grade, "Es")
            
            dcr, status = deckdesign.design_deck_slab(
            self.input_dict, fck=fck, fctm=fctm, fy=fy, Ecm=Ecm, Es=Es,
            design_results=design_results,
            bf_top_mm= resolve_girder_value(self.input_dict, KEY_MP_GIRDER_TOP_FLANGE_WIDTH),
            stud_height_mm=float(self.input_dict[KEY_DS_STUD_HEIGHT]),
            print = False
        )
            
            # check for deck slab design
            if status["overall_status"] == "FAIL":      
                return "DECKSLAB FAIL"
            
            
            # Stage 7: Transverse Member Design
            # self.crossbracing_design_results = self._run_stage("7", self._stage_transverse_design)
            
            # self.bridge_component_solver()
            # self.compute_load_effects_cache() 
            
            return "PASS"
            
        except Exception:
            
            raise

        
    """
    Objective function for minimisation of weight of Plate Girder bridge
    Returns superstructure weight in kN
    """
    def self_weight(self) -> float:

        n = self.input_dict[KEY_TS_NO_OF_GIRDERS]
        t_slab = self.input_dict[KEY_TS_DECK_THICKNESS]
        D = self.input_dict[KEY_MP_GIRDER_DEPTH]            
        
        span_length = self.input_dict[KEY_SPAN]
        deck_width= self.input_dict[KEY_TS_OVERALL_WIDTH]
        
        bf = self.input_dict[KEY_MP_GIRDER_TOP_FLANGE_WIDTH]
        tf = self.input_dict[KEY_MP_GIRDER_TOP_FLANGE_THICKNESS]
        tw = self.input_dict[KEY_MP_GIRDER_WEB_THICKNESS]
            
        section = SteelSection(D, bf, tf, bf, tf, tw)
        
        weight_of_steel    = n * section.A_steel * span_length * 25  # in kN
        weight_of_concrete = t_slab * deck_width * span_length * 78.5 # in kN
        
        return weight_of_steel + weight_of_concrete


# x = [n, t_slab, D] (all dimensions in metres)
def update_dict(inp: dict, x: np.ndarray) -> dict:
    
    fy_steel = inp[KEY_MATERIAL_GIRDER_FY]
    deck_width = inp[KEY_TS_OVERALL_WIDTH]
    epsilon = sqrt(250 / fy_steel)
    
    n = x[0]
    inp[KEY_TS_NO_OF_GIRDERS] = n   
    
    spacing = _floor(deck_width/ n, 0.01)   # in metre
    inp[KEY_TS_GIRDER_SPACING] = spacing
    
    inp[KEY_TS_DECK_THICKNESS] = x[1]
    inp[KEY_MP_GIRDER_DEPTH] = x[2]
    
    bf = 0.5 * x[2] # in mm
    inp[KEY_MP_GIRDER_TOP_FLANGE_WIDTH] = bf
    inp[KEY_MP_GIRDER_BOTTOM_FLANGE_WIDTH] = bf # in mm
    
    tf = _ceiling_plate(bf * 0.5 / (9.4 * epsilon)) 
    inp[KEY_MP_GIRDER_TOP_FLANGE_THICKNESS] = tf 
    inp[KEY_MP_GIRDER_BOTTOM_FLANGE_THICKNESS] = tf 
    
    dw = x[2] - 2 * tf
    inp[KEY_MP_GIRDER_WEB_THICKNESS] = _ceiling_plate(dw / (67 * epsilon)) 
    
    inp[KEY_TS_DECK_OVERHANG] = (deck_width - (n-1) * spacing) * 0.5
    
    # Live load vehicles
    inp[KEY_LL_IRC_CLASS_A] = True
    inp[KEY_LL_IRC_70R_TRACKED] = True
    
    # inp[KEY_MP_GIRDER_SYMMETRY] = "Symmetric"
    
    return inp

# ------------------------------------------------------------------------------
#  optimize — main entry point function for bridge parameters optimisation
#  design_vector = [no_of_girders, deck_slab_thickness, girder_depth]
# ------------------------------------------------------------------------------

def optimize_dict(inp: dict):
    
    # --------------- Initial design variables setup ------------------------
    span_length = inp[KEY_SPAN]
    deck_width = round(inp[KEY_TS_OVERALL_WIDTH], 3)
    
    n_min = math.ceil(deck_width * 10 / span_length)   # min number of girders to avoid shear lag effects on composite section
    n_min = max(2.0, n_min)
    n_max = math.floor(min(deck_width , (deck_width * 20.0 / span_length))) # min girder spacing to be provided IRC 24 Cl. 504.3
    
    t_slab_min = 150
    t_slab_max = 1000
    
    D_min = (span_length * 1000 / 25)
    D_max = math.floor(span_length * 1000 / 15)
    
    # design_vector = [no_of_girders, deck_slab_thickness, girder_depth]
    # We start with the absolute maximum design vector for optimisation
    best_arr : np.ndarray = np.array([n_max, t_slab_min, D_max])
    
    test_pgb = TrialPlateGirderBridge(inp, best_arr)
        
    print("\nRunning with ",n_max," girders")
    
    design_status = test_pgb.design_is_feasible()
    
    # Check with increased slab thickness is slab design fails
    if design_status == "DECKSLAB FAIL":
        
        while t_slab_min < t_slab_max:
            
            t_slab_min = _ceil((t_slab_min + t_slab_max) / 2 , 25)   
            best_arr[1] = t_slab_min
            
            test_pgb1 = TrialPlateGirderBridge(inp, best_arr)
            if test_pgb1.design_is_feasible() == "PASS":
                
                best_arr[1] = t_slab_min
                design_status = "PASS"
                break
            
    
    elif design_status == "GIRDER FAIL":       

        print("Optimization unsuccessful")
        best_arr = np.array([n_min, t_slab_min, D_min]) # return with absolute minimum in case fail to optimize              
                
    else:
        
        # Number of Girders Optimization
        best_n_val = n_max
        arr = best_arr.copy()
        arr[0] = n_min
        test_pgb = TrialPlateGirderBridge(inp,arr)  # check with minimum number of girders
        
        if test_pgb.design_is_feasible() == "PASS":
            best_arr[0] = n_min
        else:    
            n_min += 1
            n_max -= 1  
            while n_min <= n_max:
                
                current_n_val = math.ceil((n_max + n_min) / 2)
                arr = best_arr.copy()
                arr[0] = current_n_val
                test_pgb = TrialPlateGirderBridge(inp,arr)
                
                if test_pgb.design_is_feasible() == "PASS":   
                    best_n_val = current_n_val
                    n_max = current_n_val - 1       
                else:
                    n_min = current_n_val + 1
        
        best_arr[0] = best_n_val          
                    
    # Girder Depth Optimimsation   
    """
    if design_status == "PASS":
        
        best_depth_val = D_max
        current_depth_val = D_min
        arr = best_arr.copy()
        arr[2] = current_depth_val
        test_pgb = TrialPlateGirderBridge(inp,arr)
        
        if test_pgb.design_is_feasible() == "PASS":
            best_arr[2] = current_depth_val
        
        else:    
            while D_min <= D_max:
                current_depth_val = _ceil((D_max + D_min) / 2, 5)
                arr = best_arr.copy()
                arr[2] = _ceil(current_depth_val, 5)
                test_pgb = TrialPlateGirderBridge(inp, arr)
                
                if test_pgb.design_is_feasible() == "PASS":
                    best_depth_val = current_depth_val
                    D_max = current_depth_val - 1
                
                else:
                    D_min = current_depth_val + 1
        
        best_arr[2] = best_depth_val  
    """
    update_dict(inp, best_arr)
    # optimized_bridge_utility_ratio = optimized_bridge.utility_ratio
