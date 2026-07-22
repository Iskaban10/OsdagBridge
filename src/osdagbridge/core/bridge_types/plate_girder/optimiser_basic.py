"""
Typical usage
-------------
  optimized_dict = optimize_dict(input_dict)
  Optimizes bridge parameters in the input dictionary
"""

from __future__ import annotations

import math
import time
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
    KEY_MP_GIRDER_WEB_DEPTH,
    KEY_MATERIAL_GIRDER_FY,
    KEY_DS_STUD_HEIGHT,
)
from osdagbridge.core.bridge_types.plate_girder.designer import (
    SteelSection,
    run_design_check
)

from . import deckdesign
from .defaults import solve_extend_basic_input_dict


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
    # Smallest IS 2062 plate thickness ≥ value_mm (always structurally safe)
    for p in _STD_PLATES:
        if p >= value_mm:
            return float(p)
    return float(_STD_PLATES[-1])


def _ceil(v: float, n: float = 10.0) -> float:
    # Ceil to nearest multiple of n.
    return float(math.ceil(v / n) * n)


def _floor(v: float, n: float = 10.0) -> float:
    # Round to nearest multiple of n.
    return float(math.floor(v / n) * n)


def clamp(v: float, lo: float, hi: float) -> float:
    # Hard-clip v to [lo, hi]
    return max(lo, min(v, hi))

# TrialPlateGirderBridge which creates PlateGirderBridges with given design vectors

class TrialPlateGirderBridge(PlateGirderBridge):
    
    def __init__(self, input_dict: dict, x: np.ndarray, design_number: int = 0) -> None:
        
        super().__init__()
        update_dict(input_dict, x)
        solve_extend_basic_input_dict(self.input_dict, input_dict, optimisation = True)
        
        self.design_number = design_number
        
        self.set_input(self.input_dict)

    
    def design_is_feasible(self, check_slab = True, show = True) -> str:
        
        # Run the full Osdag grillage + IRC 22:2015 DCR pipeline.
        
        try:
            # with mute_stdout():
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
            self.results = PlateGirderAnalysisResults(dataset=dataset, bridge=self.grillage_model, edge_dist = edge_dist)
            self.report_text, self.engine, self.design_results = run_design_check(plate_girder_bridge=self, analysis_results=self.results, print_report=False)
            
            if self.engine.overall_status() == "FAIL":
                if show:
                    self.show("GIRDER FAIL")
                return "GIRDER FAIL"
            
            # Deck Slab design
            if check_slab:
                # with mute_stdout():
                concrete_grade = str(self.input_dict[KEY_DECK_CONCRETE_GRADE_BASIC]).strip()
                rebar_grade = str(self.input_dict[KEY_DS_REINF_MATERIAL]).strip()

                fck = self._lookup_material(concrete_grade, "fck")
                Ecm = self._lookup_material(concrete_grade, "Ecm")
                fctm = self._lookup_material(concrete_grade, "fctm")
                fy = self._lookup_material(rebar_grade, "fy")
                Es = self._lookup_material(rebar_grade, "Es")
                
                dcr, status = deckdesign.design_deck_slab(
                self.input_dict, fck=fck, fctm=fctm, fy=fy, Ecm=Ecm, Es=Es,
                design_results=self.design_results,
                bf_top_mm= resolve_girder_value(self.input_dict, KEY_MP_GIRDER_TOP_FLANGE_WIDTH),
                stud_height_mm=float(self.input_dict[KEY_DS_STUD_HEIGHT]),
                optimisation = True)
                    
                # check for deck slab design status
                if status["overall_status"] == "FAIL":  
                    if show:
                        self.show("DECKSLAB FAIL")  
                    return "DECKSLAB FAIL"

            if show:
                self.show("PASS")
            return "PASS"
            
        except Exception:            
            raise
        
    def show(self, status = ""):
        
        span_length = self.input_dict[KEY_SPAN]
        deck_width = round(self.input_dict[KEY_TS_OVERALL_WIDTH],3)
        slab_t = self.input_dict[KEY_TS_DECK_THICKNESS]
        n = self.input_dict[KEY_TS_NO_OF_GIRDERS]
        spacing = self.input_dict[KEY_TS_GIRDER_SPACING]
        g_depth = self.input_dict[KEY_MP_GIRDER_DEPTH]
        
        print("-" * 25)
        if self.design_number > 0:
            print("Candidate design: ", self.design_number)
        else:
            print("Candidate design ")
            
        print(f"Config: L: {span_length}m | W: {deck_width}m | {n} girders @ {spacing}m | Depth: {g_depth}mm | Slab thickness: {slab_t}mm")
        if status != "":
            print("Design Check Status: " , status)
        print("-" * 25)
        
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
    
    bf = 0.3 * x[2] # in mm
    inp[KEY_MP_GIRDER_TOP_FLANGE_WIDTH] = bf
    inp[KEY_MP_GIRDER_BOTTOM_FLANGE_WIDTH] = bf # in mm
    
    tf = _ceiling_plate(bf * 0.5 / (9.4 * epsilon)) 
    inp[KEY_MP_GIRDER_TOP_FLANGE_THICKNESS] = tf 
    inp[KEY_MP_GIRDER_BOTTOM_FLANGE_THICKNESS] = tf 
    
    dw = x[2] - 2 * tf
    inp[KEY_MP_GIRDER_WEB_THICKNESS] = _ceiling_plate(dw / (67 * epsilon)) 
    
    inp[KEY_TS_DECK_OVERHANG] = (deck_width - (n-1) * spacing) * 0.5
    
    return inp

# ------------------------------------------------------------------------------
#  optimize — main entry point function for bridge parameters optimisation
#  design_vector = [no_of_girders, deck_slab_thickness, girder_depth]
# ------------------------------------------------------------------------------

def optimize_dict(inp: dict):
    
    start_time = time.perf_counter()
    # --------------- Initial design variables setup ------------------------
    span_length = inp[KEY_SPAN]
    deck_width = round(inp[KEY_TS_OVERALL_WIDTH], 3)
    
    n_min = math.ceil(deck_width * 10 / span_length)   # min number of girders to avoid shear lag effects on composite section
    n_min = max(2.0, n_min)
    n_max = math.floor(min(deck_width , (deck_width * 20.0 / span_length))) # min girder spacing to be provided IRC 24 Cl. 504.3
    
    t_slab_min = 150
    t_slab_max = 1000
    
    D_min = (span_length * 1000 / 25)
    D_max = _floor(span_length * 1000 / 15 , 10)
    
    design_number:int = 0
    min_self_wt:float = 0
    # design_vector = [no_of_girders, deck_slab_thickness, girder_depth]
    # We start with the absolute maximum design vector for optimisation
    best_arr : np.ndarray = np.array([n_max, t_slab_min, D_max])
    print("-"*20,"\nSTARTING OPTIMISATION\n","-"*20)
    
    design_number += 1
    test_pgb = TrialPlateGirderBridge(inp, best_arr, design_number)
    min_self_wt = test_pgb.self_weight()
    design_status = test_pgb.design_is_feasible()
    
    # Check with increased slab thickness is slab design fails
    if design_status == "DECKSLAB FAIL":
        
        while t_slab_min < t_slab_max:
            
            # t_slab_min = _ceil((t_slab_min + t_slab_max) / 2 , 25)   
            best_arr[1] = t_slab_min + 25   # increment by 25 for now

            design_number += 1
            test_pgb = TrialPlateGirderBridge(inp, best_arr, design_number)
            design_status = test_pgb.design_is_feasible()
            if design_status != "DECKSLAB FAIL":
                break
    
    if design_status == "GIRDER FAIL":       
        print("Optimization unsuccessful")
        best_arr = np.array([n_min, t_slab_min, D_min]) # return with absolute minimum in case fail to optimize              
        update_dict(inp, best_arr)
        print("-"*20,"\nOPTIMISATION FINISHED\n","-"*20)
        return
    
    # Number of Girders Optimization
    best_n_val = n_max
    arr = best_arr.copy()
    arr[0] = n_min
    design_number += 1
    test_pgb = TrialPlateGirderBridge(inp, arr, design_number)  # check with minimum number of girders
    
    if test_pgb.design_is_feasible() == "PASS":
        best_n_val = n_min
        min_self_wt = test_pgb.self_weight()
    else:
        n_min += 1
        n_max -= 1  
        while n_min <= n_max:
            
            current_n_val = math.ceil((n_max + n_min) / 2)
            arr[0] = current_n_val
            design_number += 1
            test_pgb = TrialPlateGirderBridge(inp, arr, design_number)
            
            if test_pgb.design_is_feasible() == "PASS":
                min_self_wt = test_pgb.self_weight()
                best_n_val = current_n_val
                n_max = current_n_val - 1       
            else:
                n_min = current_n_val + 1
    
    best_arr[0] = best_n_val      
    print(f"\nOPTIMAL GIRDER COUNT ACHIEVED SUCCESSFULLY = {best_n_val}\n")
                
    # Girder Depth Optimimsation   
    print("\nBEGINNING GIRDER DEPTH OPTIMISATION\n")
    D_min = _ceil(max(D_min, sqrt(1-(1/best_n_val)) * D_max) , 10)
    if design_status == "PASS":
        
        best_depth_val = D_max
        arr = best_arr.copy()
        arr[2] = D_min
        design_number += 1
        test_pgb = TrialPlateGirderBridge(inp,arr, design_number)
        
        if test_pgb.design_is_feasible(check_slab = False) == "PASS":
            best_depth_val = D_min      
            min_self_wt = test_pgb.self_weight()
        else:    
            while D_min <= D_max and design_number < 10:
                
                arr = best_arr.copy()
                arr[2] = (D_max + D_min) / 2
                design_number += 1
                test_pgb = TrialPlateGirderBridge(inp, arr, design_number)
                
                if test_pgb.design_is_feasible(check_slab=False) == "PASS":
                    min_self_wt = test_pgb.self_weight()
                    best_depth_val = arr[2]
                    D_max = arr[2] - 10
                
                else:
                    D_min = arr[2] + 10
        
        best_arr[2] = math.ceil(best_depth_val)
        print(f"\nOPTIMAL GIRDER DEPTH ACHIEVED = {math.ceil(best_depth_val)}\n")  
    
    update_dict(inp, best_arr)  # updated with optimal
    
    n = inp[KEY_TS_NO_OF_GIRDERS]
    spacing = inp[KEY_TS_GIRDER_SPACING]
    slab_thk = inp[KEY_TS_DECK_THICKNESS]
    g_depth = inp[KEY_MP_GIRDER_DEPTH]
    end_time = time.perf_counter()
    if design_status == "PASS":
        print(f"OPTIMAL CANDIDATE -> {n} GIRDERS @ {spacing}m | GIRDER DEPTH: {g_depth}mm | SLAB THICKNESS: {slab_thk}mm")
        print(f"OPTIMAL CANDIDATE SUPERSTRUCTURE WEIGHT = {min_self_wt}")
        print(f"{design_number} CANDIDATE DESIGNS EVALUATED")
    
    exec_time = end_time - start_time
    print("-"*20,f"\nOPTIMISATION FINISHED AFTER {exec_time} SECONDS\n","-"*20)

