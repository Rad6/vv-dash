import asyncio
import json
from typing import Dict, List, Tuple
import logging
from collections import defaultdict, deque

from istream_player.config.config import PlayerConfig
from istream_player.core.abr import ABRController
from istream_player.core.module import Module, ModuleOption
from istream_player.models import AdaptationSet
from istream_player.models.mpd_objects import Representation
import pdb
from pyomo.environ import *
import pyomo.environ as pyo
import pyomo.opt as po
from istream_player.core.buffer import BufferManager
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.modules.analyzer.analyzer import PlaybackAnalyzer, AnalyzerSegment
# import gym
# from gym import spaces
import numpy as np
import gurobipy as gp
# import joblib as jl
import pandas as pd 
import math


@ModuleOption("360_hm", requires=[BufferManager, BandwidthMeter, PlaybackAnalyzer])
class ABR360HeadMovement(Module, ABRController):
    log = logging.getLogger("ABR360HeadMovement")
    def __init__(self, *, data: str, mode_selection: str = "heuristic"):
        super().__init__()
        self.path = data
        self.data: List[Dict] = []
        self.focusedAdapSet = []
        self.mode_selection = mode_selection
        self.previous_bw = 0
        self.previous_quality_selection = {}
        self.previous_bitrates = {}
        self.infeasible_solutions = 0
        self.quality_change = 0
        self.not_enough_bw_counter = 0
        self.prev_stall_counter = 0


    async def setup(
            self, 
            config: PlayerConfig, 
            buffer_manager: BufferManager,
            bandwidth_meter: BandwidthMeter,
            analyzer: PlaybackAnalyzer,
            **kwargs
        
    ):
        with open(self.path) as f:
            self.data = json.load(f)["data"]
        self.fov_groups_number = config.fov_groups_number
        self.resolution_ladder = config.resolution_ladder
        self.resolution_threshold = config.resolution_threshold
        self.bitrate_ladder = config.bitrate_ladder
        self.max_initial_bitrate = config.static.max_initial_bitrate
        self.panic_buffer = config.panic_buffer_level 
        self.safe_buffer = config.safe_buffer_level
        self.bw_penalty = config.bw_penalty_factor
        self.priority_weight = config.priority_weight
        self.priority_weight_tbra = config.priority_weight_tbra
        self.weight_obj_func = config.weight_obj_func
        self.bw_limit = config.bw_limit
        self.bw_error_tolerance = config.bw_error_tolerance
        self.bw_error_offset = config.bw_error_offset
        self.buffer_manager = buffer_manager
        self.bandwidth_meter = bandwidth_meter
        self.analyzer = analyzer
        self.pool_search_mode = config.pool_search_mode
        self.pool_solutions = config.pool_solutions
        self.pool_gap = config.pool_gap
        self.solver_output_mode = config.output_flag
        self.solver_mode = config.solver_mode
        self.weight_prediction = config.weight_prediction
        self.pred_model_param = config.prediction_model_parameters
        self.pred_rate_controller = config.prediction_weight_rate_controller
        self.fov_prediction_mode = config.fov_prediction_mode

    async def run(self):
        for entry in self.data:
            self.focusedAdapSet = entry["focusAdapSet"]
            await asyncio.sleep(entry["duration"])



    def fov_group_selection(self, adaptation_sets: Dict[int, AdaptationSet]) -> Dict[int, list]:
        """
        This function defines which tiles is assgined to which FoV group.

        Parameters:
        focusedAdapSet (list): A list of FoV tiles.
        adaptationSets (AdaptationSet): The adaptation sets.

        Returns:
        Dict[int, list]: A dictionary where the keys are FoV group IDs and the values are lists of assgined tiles.
        """
        fov_groups = {id: [] for id in range(1, self.fov_groups_number + 1)} #defining the FoV groups (1 is the best)

        adap_grid = {}
        for adapId, adapSet in adaptation_sets.items():
            if adapSet.srd is not None:
                adap_grid[(adapSet.srd[1], adapSet.srd[2])] = adapId
        for tile in self.focusedAdapSet:
            fov_groups[1] = self.focusedAdapSet #defining the first priority FoV group
            adaptation_set = adaptation_sets[tile]
            _, col, row, tile_cols, tile_rows, grid_cols, grid_rows = adaptation_set.srd
            number_of_tiles = grid_cols * grid_rows
            for neighbourPos in (
                (col, (row - 1) % grid_rows),  # Top
                (col, (row + 1) % grid_rows),  # Bottom
                ((col - 1) % grid_cols, row),  # Left
                ((col + 1) % grid_cols, row),  # Right
            ):  
                adapId = adap_grid[neighbourPos]
                if adapId not in fov_groups[2] and adapId not in fov_groups[1]:
                    fov_groups[2].append(adapId)
        for tile in range(0, number_of_tiles):
            if tile not in fov_groups[1] and tile not in fov_groups[2]:
                if tile not in fov_groups[3]:
                    fov_groups[3].append(tile)

        return fov_groups


    def update_selection(
        self, adaptation_sets: Dict[int, AdaptationSet], index: int, fov_groups: Dict[int, list], fov_group_id: int, scheduler_mode: str
    ) -> Dict[Tuple[int,int], int]: #-->(tile_number(adaptation_set_id), FoV_group), quality(representation_id)

        final_selections = dict()

        def has_seg_id(rep: Representation):
            for seg_id, _ in rep.segments.items():
                if seg_id == index:
                    return True
            return False

        def qualities(adaptation_set: AdaptationSet):
            # this function returns a dictionary of representations that key is the representation id and the value is the representation object
            # the representations are sorted based on the representation id (from the lowest to the highest)
            repr = [
                rep
                for rep_id, rep in adaptation_set.representations.items()
                if has_seg_id(rep)
            ]
            repr.sort(key=lambda rep: rep.id)
            return repr
        
        def quality_order_selector(representations_list: List[Representation]) -> str:
            quality_order = 'lh' #low to high
            if len(representations_list) >= 2:
                if representations_list[0].bandwidth > representations_list[1].bandwidth:
                    quality_order = 'hl' #high to low
                else:
                    quality_order = 'lh' #low to high
            else:
                self.log.error(f"Adaptation set has less than 2 representations")
            return quality_order
        
        def number_of_tiles(adaptation_sets: Dict[int, AdaptationSet]) -> int:
            """
            This function calculates the number of tiles in the grid.

            Parameters:
            adaptationSets (AdaptationSet): The adaptation sets.

            Returns:
            int: The number of tiles.
            """
            number_of_tiles = 0
            adaptation_set = adaptation_sets[0]
            _, _, _, _, _, grid_cols, grid_rows = adaptation_set.srd
            number_of_tiles = grid_cols * grid_rows
            return number_of_tiles
        
        def calculate_average_tile_bitrates(adaptation_sets: AdaptationSet):
            avg_bitrate = []
            avg_bitrates = {}
            n_tiles = number_of_tiles(adaptation_sets)

            for item in range(0, len(self.bitrate_ladder)):
                for tile in range(0, n_tiles):
                    adaptation_set = adaptation_sets[tile]
                    repr_list = qualities(adaptation_set)
                    avg_bitrate.append(repr_list[item].bandwidth)
                print(avg_bitrate)
                avg_bitrates[item] = sum(avg_bitrate)/len(avg_bitrate)
                avg_bitrate = []
            
            return avg_bitrates
        
        def res_thereshold_selector(fov_groups: Dict[int, list], left_tiles: List) -> Dict[int, int]:
            resolution_thereshold = {}
            resolution_sorted = sorted(self.resolution_ladder, reverse=True)
            for item in fov_groups[1]: #best FoV group (viewport) 
                resolution_thereshold[item] = resolution_sorted[self.resolution_threshold[0]] #best resolution 0
            for item in fov_groups[2]: #near FoV group
                resolution_thereshold[item] = resolution_sorted[self.resolution_threshold[1]] #second best resolution #1
            for item in fov_groups[3]: #out of FoV group
                resolution_thereshold[item] = resolution_sorted[self.resolution_threshold[2]] #least resolution #-1

            resolution_thereshold_selected = {i: resolution_thereshold[i] for i in left_tiles}
            return resolution_thereshold_selected
        
        def bitrate_ladder_constructor(AdapSets: AdaptationSet, left_tiles: List) -> Dict[Tuple[int, int], int]:
            # This function constructs the bitrate ladder for the model
            # The output is a dictionary where the key is the tile number and representation id and the value is the bitrate
            bitrate_ladder = {} #should be ordered from low to high
            quality_order = quality_order_selector(qualities(AdapSets[0]))
            for tile in AdapSets:
                repr_list = qualities(AdapSets[tile])
                for rep_id, rep_obj in enumerate(repr_list):
                    if quality_order == 'lh':
                        bitrate_ladder[(tile, rep_id +1)] = rep_obj.bandwidth #rep_id should start from 1 because of resoultion index in model
                    elif quality_order == 'hl':
                        bitrate_ladder[(tile, len(repr_list) - rep_id)] = rep_obj.bandwidth
            
            bitrate_ladder_selected = {(i,j): bitrate_ladder[(i,j)] for i in left_tiles for j in range(1, len(self.resolution_ladder)+1)}
                
            return bitrate_ladder_selected

        def compute_bw_penalty() -> float:
            # This function selects the bandwidth smooth factor based on the previous/available bandwidth and some other parameters
            # The output is a float number (bw_smooth_factor)
            # total_tile_bitrates = {} #sum of all tile bitrates for each representation
            # available_bw = self.bandwidth_meter.bandwidth #in bps
            # for (tile, rep_id), bitrate in bitrate_ladder.items():
            #     if rep_id not in total_tile_bitrates:
            #         total_tile_bitrates[rep_id] = 0
            #     total_tile_bitrates[rep_id] += bitrate
            
            if self.quality_change != self.infeasible_solutions:
                bw_penalty_factor = self.bw_penalty
                self.quality_change += 1
                self.log.info(f" ------ Quality has been changed! |{self.quality_change}|{self.infeasible_solutions}|-----")
                
            else:
                bw_penalty_factor = 1

            return bw_penalty_factor
            #pdb.set_trace()

        def caclulate_average_throughput(tile_segments_throughput, left_tiles, tile_numbers = number_of_tiles(adaptation_sets), default_throughput = self.max_initial_bitrate) -> Dict[int, float]:
            # This function calculates the average throughput for each tile (for the last 5 segments)
            throughput_values = defaultdict(lambda: deque(maxlen=6)) #change the maxlen to change the number of segments
            tile_list = list(range(0, tile_numbers))

            for (segment_index, tile_number), throughput in tile_segments_throughput.items():
                throughput_values[tile_number].append(throughput)
            
            average_throughput = {tile: sum(values) / len(values) for tile, values in throughput_values.items()}
            for item in tile_list:
                if item not in average_throughput:
                    average_throughput[item] = default_throughput

            average_throughput_selected = {i: average_throughput[i] for i in left_tiles}
            return average_throughput_selected
   
        def calculate_tile_weight(pred_threshold=0.8, pred_alpha=20.0, time_beta=0.5):
            """
            Calculate the total weight for a tile based on FoV zone weight, prediction accuracy weight, and prediction window weight.

            Parameters:
            W_zone (float): FoV zone weight (0 to 1).
            acc (float): Accuracy of the prediction algorithm (0 to 1).
            time (float): Length of the prediction window.
            alpha (float): Sensitivity coefficient for W_zone.
            beta (float): Sensitivity coefficient for W_pred.
            gamma (float): Sensitivity coefficient for W_time.
            pred_threshold (float): Threshold for prediction accuracy (default is 0.8).
            pred_alpha (float): Steepness control for the sigmoid function (default is 1.0).
            time_beta (float): Decay rate constant for the exponential function (default is 1.0).

            Returns:
            float: Total weight for all tiles.
            """

            alpha = self.weight_prediction[0]
            beta = self.weight_prediction[1]
            gamma = self.weight_prediction[2]

            acc = self.pred_model_param[0]
            time = self.pred_model_param[1]

            pred_threshold = self.pred_rate_controller[0]
            pred_alpha = self.pred_rate_controller[1]
            time_beta = self.pred_rate_controller[2]


            W_tiles = {}
            # Calculate W_pred using the shifted sigmoid function
            W_pred = 1 / (1 + math.exp(-pred_alpha * (acc - pred_threshold)))

            # Calculate W_time using the exponential decay function
            W_time = W_pred * math.exp(-time_beta * (1 - acc) * time)


            for tile in range(0, number_of_tiles(adaptation_sets)):
                if tile in fov_groups[1]: #best FoV group (viewport)
                    W_zone = self.priority_weight[0]
                elif tile in fov_groups[2]: #near FoV group
                    W_zone = self.priority_weight[1]
                elif tile in fov_groups[3]: #out of FoV group
                    W_zone = self.priority_weight[2]
                # Calculate the total weight for the tile
                W_tiles[tile] = alpha * W_zone + beta * W_pred + gamma * W_time

            return W_tiles
        
        def optimized_result_output(pyo_model, optimized_result, solver_status, left_tiles) -> Dict[int, int]:
            # This function returns the optimized results of the model
            # The output is a dictionary where the key is the tile number and the value is the selected resolution
            final_selections = {}
            if (optimized_result.solver.status == pyo.SolverStatus.ok) and (optimized_result.solver.termination_condition == pyo.TerminationCondition.optimal):
            # An optimal solution was found
                for i in left_tiles: #model.N
                    for j in range(1, len(self.resolution_ladder)+1): #model.M
                        if pyo.value(pyo_model.X[i,j]) !=0:
                            #self.log.info(f"{pyo_model.X[i,j]} is selected")
                            #self.log.info(f"Tile {i} with resolution {self.resolution_ladder[j-1]}")
                            final_selections[i] = j #the index of the soreted resolution ladder (low to high) if the resolution ladder is sorted from low to high
            else:
                print("No solution found!")
            return final_selections

        def gurobi_to_dict(gurobi_vars, left_tiles) -> Dict[int, int]:
            # This function converts the gurobi variables to the output dictionary keys: tile number, value: selected resolution
            results_dict = {} #key: tile number, value: the key of selected resolution + 1
            results_res = {} #key: tile number, value: selected resolution
            length_resolutions = len(self.resolution_ladder)
            var_number_list = []
            def find_key_and_index_by_value(dictionary, target_value):
                for key, values in dictionary.items():
                    if target_value in values:
                        self.log.debug(f"target_value is {target_value} --> Key: {key} - Value: {values.index(target_value)}")
                        return key, values.index(target_value)
                return None, None

            # if scheduler_mode == "all_tiles":
            #     for var in gurobi_vars:
            #         if var.Xn > 0.5:
            #             var_number = int(var.VarName[1:])
            #             var_number_list.append(var_number)
            #             tile_number = (var_number - 2) // length_resolutions
            #             res_number = (var_number - 2) % length_resolutions
            #             results_dict[tile_number] = res_number + 1 #the index of the soreted resolution ladder (low to high) if the resolution ladder is sorted from low to high
            #             results_res[tile_number] = self.resolution_ladder[res_number]
            # elif scheduler_mode == "fov_tiles":
            #     len_left_tiles = len(left_tiles)
            tile_gurobi_map = {}
            for index, item in enumerate(left_tiles):
                tile_gurobi_map[item] = list(range(index*length_resolutions, index*length_resolutions + length_resolutions))
            
            for var in gurobi_vars: 
                if var.Xn > 0.5:
                    var_number = int(var.VarName[1:])
                    var_number_list.append(var_number)
                    if find_key_and_index_by_value(tile_gurobi_map, var_number - 2 ) is not None:
                        tile_number, res_number = find_key_and_index_by_value(tile_gurobi_map, var_number - 2)
                        results_dict[tile_number] = res_number + 1
                        results_res[tile_number] = self.resolution_ladder[res_number]

            # pdb.set_trace()
            # Check if results_dict has all tiles
            # expected_tiles = set(range(0, number_of_tiles(adaptation_sets)))
            # if not expected_tiles.issubset(results_dict.keys()):
            #     raise ValueError(f"results_dict does not contain all tiles - {results_dict.keys()} - {var_number_list} ")
            return results_dict, results_res
        
        def best_solution_selector(best_solution, gurobi_solution, best_solution_dict, gurobi_result_dict, fov_groups, fov_group_id) -> Dict[int, int]:
            # This function selects the best solution among the found solutions
            # The FoV group with the highest priority is selected then near FoV and out of FoV
            if best_solution == gurobi_solution:
                # self.log.info(f"Both solutions are the same")
                return best_solution_dict, best_solution
            else:
                if fov_group_id == 1:
                    current_fov = fov_groups[1]
                    near_fov = fov_groups[2]
                elif fov_group_id == 2:
                    current_fov = fov_groups[2]
                    near_fov = fov_groups[3]
                elif fov_group_id == 3:
                    current_fov = fov_groups[3]
                    near_fov = fov_groups[3]
                # pdb.set_trace()
                # self.log.info(f"current fov: {current_fov} - near fov: {near_fov} - best solution: {best_solution} - gurobi solution: {gurobi_solution}")
                
                sum_res_tile_fov_best = sum([best_solution[i] for i in current_fov])
                sum_res_tiles_fov_gurobi = sum([gurobi_solution[i] for i in current_fov])
                sum_res_near_fov_best = sum([best_solution[i] for i in near_fov])
                sum_res_near_fov_gurobi = sum([gurobi_solution[i] for i in near_fov])

                if sum_res_tile_fov_best < sum_res_tiles_fov_gurobi:
                    best_solution = gurobi_solution
                    best_solution_dict = gurobi_result_dict
                elif sum_res_tile_fov_best == sum_res_tiles_fov_gurobi:
                    if sum_res_near_fov_best < sum_res_near_fov_gurobi:
                        best_solution = gurobi_solution
                        best_solution_dict = gurobi_result_dict
                    elif sum_res_near_fov_best == sum_res_near_fov_gurobi:
                        if sum(best_solution.values()) < sum(gurobi_solution.values()):
                            best_solution = gurobi_solution
                            best_solution_dict = gurobi_result_dict

            return best_solution_dict, best_solution

        def left_tiles_selector(fov_groups: Dict[int, list], fov_group_id:int ) -> list:
            # This function selects the tiles that are not yet processed in the model (fov_tiles mode)
            # left_tiles: the tiles that are not yet processed 
            # left_tiles_selected: the tiles that will be requested at the current iteration (FoV group)
            left_tiles = []
            left_tiles_selected = []
            if fov_group_id == 1: 
                left_tiles = fov_groups[1] + fov_groups[2] + fov_groups[3]
                left_tiles_selected = fov_groups[1]
            elif fov_group_id == 2:
                left_tiles = fov_groups[2] + fov_groups[3]
                left_tiles_selected = fov_groups[2]
            elif fov_group_id == 3: 
                left_tiles = fov_groups[3]
                left_tiles_selected = fov_groups[3]
            
            left_tiles.sort()
            left_tiles_selected.sort()
            return left_tiles, left_tiles_selected

        def gwo(pop_size, dim, bounds, max_iter, evel_func):
            population = np.array([np.random.uniform(bounds[d][0], bounds[d][1], pop_size) for d in range(dim)]).T
            fitness = np.apply_along_axis(evel_func, 1, population)

            # Indentify alpha, beta, and delta wolves
            alpha_idx = np.argmin(fitness)
            alpha, alpha_fit = population[alpha_idx], fitness[alpha_idx]

            beta_idx = np.argsort(fitness)[1]
            beta, beta_fit = population[beta_idx], fitness[beta_idx]

            delta_idx = np.argsort(fitness)[2]
            delta, delta_fit = population[delta_idx], fitness[delta_idx]

            for t in range(max_iter):
                for i in range(pop_size):
                    for j in range(dim):
                        r1, r2 = np.random.rand(), np.random.rand()
                        A1, C1 = 2 * r1 - 1, 2 * r2
                        D_alpha = np.abs(C1 * alpha[j] - population[i, j])
                        X1 = alpha[j] - A1 * D_alpha

                        r1, r2 = np.random.rand(), np.random.rand()
                        A2, C2 = 2 * r1 - 1, 2 * r2
                        D_beta = np.abs(C2 * beta[j] - population[i, j])
                        X2 = beta[j] - A2 * D_beta

                        r1, r2 = np.random.rand(), np.random.rand()
                        A3, C3 = 2 * r1 - 1, 2 * r2
                        D_delta = np.abs(C3 * delta[j] - population[i, j])
                        X3 = delta[j] - A3 * D_delta

                        # update the position and clamp to the bounds
                        population[i, j] = np.clip((X1 + X2 + X3) / 3, bounds[j][0], bounds[j][1])
                
                print(f"Iteration {t}: Best Fitness: {alpha_fit}")
                print(f"population: {population}")
                fitness = np.apply_along_axis(evel_func, 1, population)

                alpha_idx = np.argmin(fitness)
                alpha, alpha_fit = population[alpha_idx], fitness[alpha_idx]

                beta_idx = np.argsort(fitness)[1]
                beta, beta_fit = population[beta_idx], fitness[beta_idx]

                delta_idx = np.argsort(fitness)[2]
                delta, delta_fit = population[delta_idx], fitness[delta_idx]
                
                print(f"Iteration {t}: Best Fitness: {alpha_fit}")

            return alpha, alpha_fit
        
        def quality_selection_optimized(adaptation_sets: Dict[int, AdaptationSet], fov_groups: Dict[int, list]):
            
            # --------- Inputs -----------
            FoV = fov_groups[1]
            near_FoV = fov_groups[2]
            mod_scheduler = scheduler_mode #inputs of scheduler and update_selection
            left_tiles, left_tiles_selected = left_tiles_selector(fov_groups, fov_group_id)

            n_tiles = number_of_tiles(adaptation_sets)
            n_res = len(self.resolution_ladder)
            resolutions = self.resolution_ladder
            bitrate_ladder = bitrate_ladder_constructor(adaptation_sets, left_tiles)
            bw_penalty_factor = compute_bw_penalty()
            seg_duration = 1
            #self.log.info(f"Bandwidth Penalty Factor is: {bw_penalty_factor}")

            # Initialize previous_qualities with keys from tile 0 to n_tiles and values of 1 (lowest quality)
            previous_qualities = {i: 1 for i in range(0, n_tiles)}
            previous_bitrates = {}
            if mod_scheduler == "all_tiles":
                if self.previous_quality_selection:
                    previous_qualities = self.previous_quality_selection
                total_previous_bitrates = 0
                seg_duration = 1
            elif mod_scheduler == "fov_tiles":
                if not self.previous_quality_selection: #if the previous quality selection is empty
                    self.previous_quality_selection = {i: 1 for i in range(0, n_tiles)}
                else:
                    if fov_group_id == 1:
                        previous_qualities = self.previous_quality_selection
                        previous_bitrates = self.previous_bitrates
                        seg_duration = 1
                    else:
                        left_tiles_prev, left_tiles_prev_selected = left_tiles_selector(fov_groups, fov_group_id -1)
                        for tile in left_tiles_prev_selected:
                            previous_qualities[tile] = self.previous_quality_selection[tile]
                            previous_bitrates[tile] = self.previous_bitrates[tile]
                        seg_duration = 0.33
                total_previous_bitrates = sum(previous_bitrates.values())
            
            previous_qualities = {key: resolutions[value - 1] for key, value in previous_qualities.items()}
            self.log.info(f"Previous Qualities are: {previous_qualities}")  
            self.log.info(f"Previous Bitrates are: {previous_bitrates} - Total Bitrate: {total_previous_bitrates}") 
     

        #if available_bw > self.previous_bw: #this can be changed based on the aggressiveness
            #self.log.debug(f"Re-optimizing the quality selection")
            # --------- Model -----------
            model = pyo.ConcreteModel()

            if mod_scheduler == "all_tiles":
                model.N = pyo.RangeSet(0, n_tiles - 1) #tile number starts from 0
                model.N_quality = pyo.RangeSet(0, n_tiles - 1)
            elif mod_scheduler == "fov_tiles":
                model.N = pyo.Set(initialize = left_tiles)
                model.N_quality = pyo.RangeSet(0, n_tiles - 1) 
            model.M = pyo.RangeSet(1, n_res) #resolution number starts from 1
            if self.bandwidth_meter.bandwidth <= self.bw_limit * self.bw_error_tolerance: #25% error in the bandwidth estimation is acceptable
                model.BW = self.bandwidth_meter.bandwidth #in bps
            else:
                model.BW = self.bw_limit * self.bw_error_offset # 10% offset for the bandwidth estimation
                # if scheduler_mode == "all_tiles":
                #     model.BW = self.bandwidth_meter.bandwidth
                # elif scheduler_mode == "fov_tiles":
                #     if fov_group_id == 2 or fov_group_id == 3:
                #         model.BW = self.bandwidth_meter.bandwidth/3
                #         total_previous_bitrates = total_previous_bitrates/3
                #     elif fov_group_id == 1:
                #         model.BW = self.bandwidth_meter.bandwidth
            model.buffer_level = self.buffer_manager.buffer_level

            self.log.info(f"Available Bandwidth is: {model.BW/1000000} Mbps")
            self.log.info(f"buffer level is: {self.buffer_manager.buffer_level}")
            #self.log.info(f"segment throughput is: {caclulate_average_throughput(self.analyzer.get_tile_segment_throughput())}")
            #self.log.info(f"Bitrate Ladder is: {bitrate_ladder}")
            self.log.info(f"previous dl_times are: {sum(caclulate_average_throughput(self.analyzer.get_tile_segment_dl_time(), left_tiles).values())}")

            model.RES = pyo.Param(model.M, initialize = {i+1: res for i, res in enumerate(resolutions)}) #default order: low - high
            model.prev_qualities = pyo.Param(model.N_quality, initialize=previous_qualities)
            model.bitrate = pyo.Param(model.N, model.M, initialize = bitrate_ladder)
            if mod_scheduler == "fov_tiles":
                if fov_group_id == 1:
                    self.resolution_therehold = [0, 1, -1]
                elif fov_group_id == 2:
                    self.resolution_thereshold = [-1, 0, -1]
                elif fov_group_id == 3:
                    self.resolution_thereshold = [-1, -1, 0]
            model.RES_th = pyo.Param(model.N, initialize = res_thereshold_selector(fov_groups, left_tiles))
            model.avg_throughput = pyo.Param(model.N, initialize = caclulate_average_throughput(self.analyzer.get_tile_segment_throughput(), left_tiles))
            model.avg_dl_time = pyo.Param(model.N, initialize = caclulate_average_throughput(self.analyzer.get_tile_segment_dl_time(), left_tiles))
            #model.VMAF = pyo.Param(model.RES, model.bitrates, initialize = VMAF)

            model.X = pyo.Var(model.N, model.M, within = pyo.Boolean)

            
            if self.fov_prediction_mode == False:
                #default objective function
                def obj_rule(model):
                    return (sum(
                        model.X[i,j] * model.RES[j] * (
                            self.priority_weight[0] if i in FoV else self.priority_weight[1] if i in near_FoV else self.priority_weight[2]
                        ) 
                        for j in model.M for i in model.N)
                        )
                model.obj = pyo.Objective(sense=pyo.maximize, rule=obj_rule)

            elif self.fov_prediction_mode == True:
                # considering FoV prediction models 
                self.log.info(f"Prediction Weights are: {calculate_tile_weight()}")
                model.tile_weight = pyo.Param(model.N, initialize=calculate_tile_weight())

                def obj_rule_prediction(model):
                    return (sum(
                        model.X[i,j] * model.RES[j] * model.tile_weight[i]
                        for j in model.M for i in model.N)
                    )
                model.obj = pyo.Objective(sense=pyo.maximize, rule=obj_rule_prediction)
            

            def const1 (model, i): #each tile is assigned to only one resolution
                return sum(model.X[i,j] for j in model.M) == 1
            model.const1 = pyo.Constraint(model.N, rule = const1)

            def const2 (model, i): #according to the FoV group, assign the resolution
                if i in FoV:  # Check for tiles in the FoV
                    return sum(model.RES[j]* model.X[i,j] for j in model.M) <= model.RES_th[i]
                elif i in near_FoV:  # Check for tiles in the near FoV
                    return sum(model.RES[j]* model.X[i,j] for j in model.M) <= model.RES_th[i]
                else:  # Check for tiles out of the FoV
                    return sum(model.RES[j]* model.X[i,j] for j in model.M) <= model.RES_th[i]
            model.const2 = pyo.Constraint(model.N, rule = const2)

            def const3 (model, i): #FoV tiles should have better equal quality than their previous qualities (quality switch)
                if i in FoV:  # Only check for tiles in the FoV
                    return model.prev_qualities[i] <= sum(model.RES[j]* model.X[i,j] for j in model.M)
                else:
                    return pyo.Constraint.Skip
            model.const3 = pyo.Constraint(model.N, rule = const3)

            # def const3a(model, i, j):
            #     return model.G_i[i] >= model.X[i, j] * model.RES[j] - model.prev_qualities[i]
            # model.const3a = pyo.Constraint(model.N, model.M, rule=const3a)

            # def const3b(model, i, j):
            #     return  model.G_i[i] >= model.prev_qualities[i] - model.X[i, j] * model.RES[j]
            # model.const3b = pyo.Constraint(model.N, model.M, rule=const3b)
                

            def const4 (model): #bandwidth constraint
                return sum(model.X[i,j] * model.bitrate[i, j] for j in model.M for i in model.N) <= (bw_penalty_factor * model.BW) - total_previous_bitrates #total_previous_bitrates only applied for fov_tiles mode
            model.const4 = pyo.Constraint(rule=const4)

            # def const5 (model): #buffer constraint
            #     return self.panic_buffer <= model.buffer_level - sum(model.X[i,j] * model.bitrate[i, j] / model.avg_throughput[i] for j in model.M for i in model.N) + seg_duration #seg_duration is the duration of the segment/tile 
            # model.const5 = pyo.Constraint(rule=const5)

            def const5 (model): #buffer constraint
                return self.panic_buffer <= model.buffer_level - sum(model.X[i,j] * model.avg_dl_time[i] for j in model.M for i in model.N) + seg_duration #seg_duration is the duration of the segment/tile 
            model.const5 = pyo.Constraint(rule=const5)
            # model.pprint()
            # pdb.set_trace()

            if self.solver_mode == "pyomo":
                solver = pyo.SolverFactory('gurobi')
                results = solver.solve(model, tee=False) #displaying the solver output --> tee=True

                if results.solver.termination_condition == TerminationCondition.infeasible:
                    self.log.info(f"The solution is infeasible. ---- Bandwidth is {model.BW}")
                    self.infeasible_solutions += 1
                    optimized_quality_selections = {}
                    solution_count = 0
                elif results.solver.termination_condition == TerminationCondition.optimal:
                    optimized_results = optimized_result_output(model, results, solver, left_tiles)
                    self.log.info(f"Optimized Results are: {optimized_results}") #tile number and selected resolution index (starts from 1) (from self.resolution_ladder)
                    optimized_quality_selections = {}
                    solution_count = 1


            elif self.solver_mode == "gurobi":
                #model.write('model.lp', io_options={'symbolic_solver_labels': True})
                model.write('model.lp')
                gurobi_model = gp.read('model.lp')

                gurobi_model.Params.PoolSearchMode = self.pool_search_mode
                gurobi_model.Params.PoolSolutions = self.pool_solutions
                gurobi_model.Params.PoolGap = self.pool_gap
                gurobi_model.Params.OutputFlag = self.solver_output_mode

                # pdb.set_trace()

                gurobi_model.optimize()

                solution_count = gurobi_model.SolCount
                self.log.info(f"Number of solutions found: {solution_count}")

                if solution_count > 0:
                    best_solution = {i: self.resolution_ladder[0] for i in range(0, n_tiles)}
                    best_sol_dict = {i: 1 for i in range(0, n_tiles)}
                    for sol_index in range(solution_count):
                        self.log.debug(f"Solution {sol_index + 1}")
                        self.log.debug(f"Objective Value: {gurobi_model.objVal}")
                        results_dict, results_res = gurobi_to_dict(gurobi_model.getVars(), left_tiles)
                        self.log.debug(f"Gurobi Results: {results_res}")
                        best_sol_dict, best_solution = best_solution_selector(best_solution, results_res, best_sol_dict, results_dict, fov_groups, fov_group_id)
                        self.log.debug(f"Best Solution so far: {best_solution}")
                        self.log.debug(f"---------------------")

                    # optimized_results = optimized_result_output(model, results, solver, left_tiles)
                    optimized_results = best_sol_dict
                    self.log.info(f"Best Optimized Results are: {best_solution}") #tile number and selected resolution index (starts from 1) (from self.resolution_ladder)
                    self.log.info(f"Best Solution Dict is: {optimized_results}")
                    optimized_quality_selections = {}
                
                elif solution_count == 0:
                    self.log.info(f"The solution is infeasible. ---- Bandwidth is {model.BW}")
                    self.infeasible_solutions += 1
                    optimized_quality_selections = {}

            if mod_scheduler == "all_tiles" and solution_count > 0:
                    self.previous_quality_selection = optimized_results
                    for fov_group, tiles in fov_groups.items():  
                        for tile in tiles:
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh': #index - 1 because the index starts from 1
                                optimized_quality_selections[(tile,fov_group)] = repr_list[optimized_results[tile] - 1].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[optimized_results[tile] - 1].height} and bitrate {repr_list[optimized_results[tile] - 1].bandwidth}")
                            elif quality_order == 'hl':
                                optimized_quality_selections[(tile,fov_group)] = repr_list[len(self.resolution_ladder) - optimized_results[tile]].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - optimized_results[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - optimized_results[tile]].bandwidth}")
            
            elif mod_scheduler == "fov_tiles" and solution_count > 0:
                    for tile in left_tiles_selected:
                            self.previous_quality_selection[tile] = optimized_results[tile]
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh': #index - 1 because the index starts from 1 in the model
                                optimized_quality_selections[(tile,fov_group_id)] = repr_list[optimized_results[tile] - 1].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[optimized_results[tile] - 1].height} and bitrate {repr_list[optimized_results[tile] - 1].bandwidth}")
                                self.previous_bitrates[tile] = repr_list[optimized_results[tile] - 1].bandwidth
                            elif quality_order == 'hl':
                                optimized_quality_selections[(tile,fov_group_id)] = repr_list[len(self.resolution_ladder) - optimized_results[tile]].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - optimized_results[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - optimized_results[tile]].bandwidth}")
                                self.previous_bitrates[tile] = repr_list[len(self.resolution_ladder) - optimized_results[tile]].bandwidth

            if mod_scheduler == "all_tiles" and solution_count == 0 :
                infeasible_qualities = {i: 1 for i in range(0, n_tiles)}
                self.previous_quality_selection = infeasible_qualities
                for fov_group, tiles in fov_groups.items():
                    for tile in tiles:
                        adaptation_set = adaptation_sets[tile]
                        repr_list = qualities(adaptation_set)
                        quality_order = quality_order_selector(repr_list)
                        if quality_order == 'lh':
                            optimized_quality_selections[(tile,fov_group)] = repr_list[0].id
                            self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                        elif quality_order == 'hl':
                            optimized_quality_selections[(tile,fov_group)] = repr_list[-1].id
                            self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")
            elif mod_scheduler == "fov_tiles" and solution_count == 0 :
                for tile in left_tiles_selected:
                        self.previous_quality_selection[tile] = 1
                        adaptation_set = adaptation_sets[tile]
                        repr_list = qualities(adaptation_set)
                        quality_order = quality_order_selector(repr_list)
                        if quality_order == 'lh':
                            optimized_quality_selections[(tile,fov_group_id)] = repr_list[0].id
                            self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                            self.previous_bitrates[tile] = repr_list[0].bandwidth
                        elif quality_order == 'hl':
                            optimized_quality_selections[(tile,fov_group_id)] = repr_list[-1].id
                            self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")
                            self.previous_bitrates[tile] = repr_list[0].bandwidth 

# This part is being used when pyomo solver is used
            # if results.solver.termination_condition == TerminationCondition.infeasible:
            #     self.log.info(f"The solution is infeasible. ---- Bandwidth is {model.BW}")
            #     self.infeasible_solutions += 1
            #     optimized_quality_selections = {}
            #     # give the least quality to all tiles
            #     if mod_scheduler == "all_tiles":
            #         infeasible_qualities = {i: 1 for i in range(0, n_tiles)}
            #         self.previous_quality_selection = infeasible_qualities
            #         for fov_group, tiles in fov_groups.items():
            #             for tile in tiles:
            #                 adaptation_set = adaptation_sets[tile]
            #                 repr_list = qualities(adaptation_set)
            #                 quality_order = quality_order_selector(repr_list)
            #                 if quality_order == 'lh':
            #                     optimized_quality_selections[(tile,fov_group)] = repr_list[0].id
            #                     self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
            #                 elif quality_order == 'hl':
            #                     optimized_quality_selections[(tile,fov_group)] = repr_list[-1].id
            #                     self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")
            #     elif mod_scheduler == "fov_tiles":
            #         for tile in left_tiles_selected:
            #                 self.previous_quality_selection[tile] = 1
            #                 adaptation_set = adaptation_sets[tile]
            #                 repr_list = qualities(adaptation_set)
            #                 quality_order = quality_order_selector(repr_list)
            #                 if quality_order == 'lh':
            #                     optimized_quality_selections[(tile,fov_group_id)] = repr_list[0].id
            #                     self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
            #                     self.previous_bitrates[tile] = repr_list[0].bandwidth
            #                 elif quality_order == 'hl':
            #                     optimized_quality_selections[(tile,fov_group_id)] = repr_list[-1].id
            #                     self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")
            #                     self.previous_bitrates[tile] = repr_list[0].bandwidth 

            #     # self.log.info(f"Infeasible Solutions are: {optimized_quality_selections}")

            # elif results.solver.termination_condition == TerminationCondition.optimal:
            #     # solver status is optimal
            #     optimized_results = optimized_result_output(model, results, solver, left_tiles)
            #     self.log.info(f"Optimized Results are: {optimized_results}") #tile number and selected resolution index (starts from 1) (from self.resolution_ladder)
            #     optimized_quality_selections = {}

            #     if mod_scheduler == "all_tiles":
            #         self.previous_quality_selection = optimized_results
            #         for fov_group, tiles in fov_groups.items():  
            #             for tile in tiles:
            #                 adaptation_set = adaptation_sets[tile]
            #                 repr_list = qualities(adaptation_set)
            #                 quality_order = quality_order_selector(repr_list)
            #                 if quality_order == 'lh': #index - 1 because the index starts from 1
            #                     optimized_quality_selections[(tile,fov_group)] = repr_list[optimized_results[tile] - 1].id
            #                     self.log.info(f"Tile {tile} with resolution {repr_list[optimized_results[tile] - 1].height} and bitrate {repr_list[optimized_results[tile] - 1].bandwidth}")
            #                 elif quality_order == 'hl':
            #                     optimized_quality_selections[(tile,fov_group)] = repr_list[len(self.resolution_ladder) - optimized_results[tile]].id
            #                     self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - optimized_results[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - optimized_results[tile]].bandwidth}")
            #     elif mod_scheduler == "fov_tiles":
            #         for tile in left_tiles_selected:
            #                 self.previous_quality_selection[tile] = optimized_results[tile]
            #                 adaptation_set = adaptation_sets[tile]
            #                 repr_list = qualities(adaptation_set)
            #                 quality_order = quality_order_selector(repr_list)
            #                 if quality_order == 'lh': #index - 1 because the index starts from 1 in the model
            #                     optimized_quality_selections[(tile,fov_group_id)] = repr_list[optimized_results[tile] - 1].id
            #                     self.log.info(f"Tile {tile} with resolution {repr_list[optimized_results[tile] - 1].height} and bitrate {repr_list[optimized_results[tile] - 1].bandwidth}")
            #                     self.previous_bitrates[tile] = repr_list[optimized_results[tile] - 1].bandwidth
            #                 elif quality_order == 'hl':
            #                     optimized_quality_selections[(tile,fov_group_id)] = repr_list[len(self.resolution_ladder) - optimized_results[tile]].id
            #                     self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - optimized_results[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - optimized_results[tile]].bandwidth}")
            #                     self.previous_bitrates[tile] = repr_list[len(self.resolution_ladder) - optimized_results[tile]].bandwidth


            if fov_group_id == 3:
                self.previous_bitrates = {}

            # else:
            #     self.log.debug(f"no reoptimization is needed due to the bandwidth decrease")
            #     self.log.debug(f"previous optimized quality selections are: {self.previous_quality_selections}")
            #     optimized_quality_selections = self.previous_quality_selections    
                
            # self.previous_bw = available_bw
            # self.log.debug(f"Previous Bandwidth is: {self.previous_bw}")
            # self.log.info(f"Optimized Previous Quality Selections are: {self.previous_quality_selection}")
            #self.log.info(f"Optimized Quality Selections are: {optimized_quality_selections}")
            return optimized_quality_selections
        
        def ga_weight_selector(adaptation_sets: AdaptationSet, fov_groups: Dict[int, list], fov_group_id:int, scheduler_mode:str) -> Dict[int, int]:
            #impleneting a Genetic Algorithm to find the best weights for the optimization model objective function

            def evalute_priority_weight(weights):
                self.priority_weight[0], self.priority_weight[1], self.priority_weight[2] = weights

                FoV = fov_groups[1]
                near_FoV = fov_groups[2]
                mod_scheduler = scheduler_mode

                left_tiles, left_tiles_selected = left_tiles_selector(fov_groups, fov_group_id)

                n_tiles = number_of_tiles(adaptation_sets)
                n_res = len(self.resolution_ladder)
                resolutions = self.resolution_ladder
                bitrate_ladder = bitrate_ladder_constructor(adaptation_sets, left_tiles)
                bw_penalty_factor = compute_bw_penalty()
                seg_duration = 1

                previous_qualities = {i: 1 for i in range(0, n_tiles)}
                previous_bitrates = {}
                if mod_scheduler == "all_tiles":
                    if self.previous_quality_selection:
                        previous_qualities = self.previous_quality_selection
                    total_previous_bitrates = 0
                    seg_duration = 1
                elif mod_scheduler == "fov_tiles":
                    if not self.previous_quality_selection:
                        self.previous_quality_selection = {i: 1 for i in range(0, n_tiles)}
                    else:
                        if fov_group_id == 1:
                            previous_qualities = self.previous_quality_selection
                            previous_bitrates = self.previous_bitrates
                            seg_duration = 1
                        else:
                            left_tiles_prev, left_tiles_prev_selected = left_tiles_selector(fov_groups, fov_group_id - 1)
                            for tile in left_tiles_prev_selected:
                                previous_qualities[tile] = self.previous_quality_selection[tile]
                                previous_bitrates[tile] = self.previous_bitrates[tile]
                            seg_duration = 0.33
                    total_previous_bitrates = sum(previous_bitrates.values())

                previous_qualities = {key: resolutions[value - 1] for key, value in previous_qualities.items()}

                # --------- Model -----------
                model = pyo.ConcreteModel()
                if mod_scheduler == "all_tiles":
                    model.N = pyo.RangeSet(0, n_tiles - 1)
                    model.N_quality = pyo.RangeSet(0, n_tiles - 1)
                elif mod_scheduler == "fov_tiles":
                    model.N = pyo.Set(initialize=left_tiles)
                    model.N_quality = pyo.RangeSet(0, n_tiles - 1)
                model.M = pyo.RangeSet(1, n_res)
                if self.bandwidth_meter.bandwidth <= self.bw_limit * self.bw_error_tolerance:
                    model.BW = self.bandwidth_meter.bandwidth
                else:
                    model.BW = self.bw_limit * 1.1
                model.buffer_level = self.buffer_manager.buffer_level

                model.RES = pyo.Param(model.M, initialize={i + 1: res for i, res in enumerate(resolutions)})
                model.prev_qualities = pyo.Param(model.N_quality, initialize=previous_qualities)
                model.bitrate = pyo.Param(model.N, model.M, initialize=bitrate_ladder)
                model.RES_th = pyo.Param(model.N, initialize=res_thereshold_selector(fov_groups, left_tiles))
                model.avg_throughput = pyo.Param(model.N, initialize=caclulate_average_throughput(self.analyzer.get_tile_segment_throughput(), left_tiles))
                model.avg_dl_time = pyo.Param(model.N, initialize=caclulate_average_throughput(self.analyzer.get_tile_segment_dl_time(), left_tiles))
                model.X = pyo.Var(model.N, model.M, within=pyo.Boolean)

                def obj_rule(model):
                    return (sum(
                        model.X[i, j] * model.RES[j] * (
                            self.priority_weight[0] if i in FoV else self.priority_weight[1] if i in near_FoV else self.priority_weight[2]
                        )
                        for j in model.M for i in model.N)
                    )
                model.obj = pyo.Objective(sense=pyo.maximize, rule=obj_rule)

                def const1(model, i):
                    return sum(model.X[i, j] for j in model.M) == 1
                model.const1 = pyo.Constraint(model.N, rule=const1)

                def const2(model, i):
                    if i in FoV:
                        return sum(model.RES[j] * model.X[i, j] for j in model.M) <= model.RES_th[i]
                    elif i in near_FoV:
                        return sum(model.RES[j] * model.X[i, j] for j in model.M) <= model.RES_th[i]
                    else:
                        return sum(model.RES[j] * model.X[i, j] for j in model.M) <= model.RES_th[i]
                model.const2 = pyo.Constraint(model.N, rule=const2)

                def const3(model, i):
                    if i in FoV:
                        return model.prev_qualities[i] <= sum(model.RES[j] * model.X[i, j] for j in model.M)
                    else:
                        return pyo.Constraint.Skip
                model.const3 = pyo.Constraint(model.N, rule=const3)

                def const4(model):
                    return sum(model.X[i, j] * model.bitrate[i, j] for j in model.M for i in model.N) <= (
                            bw_penalty_factor * model.BW) - total_previous_bitrates
                model.const4 = pyo.Constraint(rule=const4)

                def const5(model):
                    return self.panic_buffer <= model.buffer_level - sum(
                        model.X[i, j] * model.avg_dl_time[i] for j in model.M for i in model.N) + seg_duration
                model.const5 = pyo.Constraint(rule=const5)

                if self.solver_mode == "pyomo":
                    solver = pyo.SolverFactory('gurobi')
                    results = solver.solve(model, tee=False) #displaying the solver output --> tee=True

                    if results.solver.termination_condition == TerminationCondition.infeasible:
                        self.log.info(f"The solution is infeasible. ---- Bandwidth is {model.BW}")
                        self.infeasible_solutions += 1
                        optimized_quality_selections = {}
                        solution_count = 0
                    elif results.solver.termination_condition == TerminationCondition.optimal:
                        optimized_results = optimized_result_output(model, results, solver, left_tiles)
                        self.log.info(f"Optimized Results are: {optimized_results}") #tile number and selected resolution index (starts from 1) (from self.resolution_ladder)
                        optimized_quality_selections = {}
                        solution_count = 1


                elif self.solver_mode == "gurobi":
                    #model.write('model.lp', io_options={'symbolic_solver_labels': True})
                    model.write('model.lp')
                    gurobi_model = gp.read('model.lp')

                    gurobi_model.Params.PoolSearchMode = self.pool_search_mode
                    gurobi_model.Params.PoolSolutions = self.pool_solutions
                    gurobi_model.Params.PoolGap = self.pool_gap
                    gurobi_model.Params.OutputFlag = self.solver_output_mode

                    gurobi_model.optimize()

                    solution_count = gurobi_model.SolCount
                    self.log.info(f"Number of solutions found: {solution_count}")

                    if solution_count > 0:
                        best_solution = {i: self.resolution_ladder[0] for i in range(0, n_tiles)}
                        best_sol_dict = {i: 1 for i in range(0, n_tiles)}
                        for sol_index in range(solution_count):
                            self.log.debug(f"Solution {sol_index + 1}")
                            self.log.debug(f"Objective Value: {gurobi_model.objVal}")
                            results_dict, results_res = gurobi_to_dict(gurobi_model.getVars(), left_tiles)
                            self.log.debug(f"Gurobi Results: {results_res}")
                            best_sol_dict, best_solution = best_solution_selector(best_solution, results_res, best_sol_dict, results_dict, fov_groups, fov_group_id)
                            self.log.debug(f"Best Solution so far: {best_solution}")
                            self.log.debug(f"---------------------")

                        # optimized_results = optimized_result_output(model, results, solver, left_tiles)
                        optimized_results = best_sol_dict
                        self.log.info(f"Best Optimized Results are: {best_solution}") #tile number and selected resolution index (starts from 1) (from self.resolution_ladder)
                        self.log.info(f"Best Solution Dict is: {optimized_results}")
                        optimized_quality_selections = {}
                    
                    elif solution_count == 0:
                        self.log.info(f"The solution is infeasible. ---- Bandwidth is {model.BW}")
                        self.infeasible_solutions += 1
                        optimized_quality_selections = {}

                if mod_scheduler == "all_tiles" and solution_count > 0:
                        self.previous_quality_selection = optimized_results
                        for fov_group, tiles in fov_groups.items():  
                            for tile in tiles:
                                adaptation_set = adaptation_sets[tile]
                                repr_list = qualities(adaptation_set)
                                quality_order = quality_order_selector(repr_list)
                                if quality_order == 'lh': #index - 1 because the index starts from 1
                                    optimized_quality_selections[(tile,fov_group)] = repr_list[optimized_results[tile] - 1].id
                                    self.log.info(f"Tile {tile} with resolution {repr_list[optimized_results[tile] - 1].height} and bitrate {repr_list[optimized_results[tile] - 1].bandwidth}")
                                elif quality_order == 'hl':
                                    optimized_quality_selections[(tile,fov_group)] = repr_list[len(self.resolution_ladder) - optimized_results[tile]].id
                                    self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - optimized_results[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - optimized_results[tile]].bandwidth}")
                
                elif mod_scheduler == "fov_tiles" and solution_count > 0:
                        for tile in left_tiles_selected:
                                self.previous_quality_selection[tile] = optimized_results[tile]
                                adaptation_set = adaptation_sets[tile]
                                repr_list = qualities(adaptation_set)
                                quality_order = quality_order_selector(repr_list)
                                if quality_order == 'lh': #index - 1 because the index starts from 1 in the model
                                    optimized_quality_selections[(tile,fov_group_id)] = repr_list[optimized_results[tile] - 1].id
                                    self.log.info(f"Tile {tile} with resolution {repr_list[optimized_results[tile] - 1].height} and bitrate {repr_list[optimized_results[tile] - 1].bandwidth}")
                                    self.previous_bitrates[tile] = repr_list[optimized_results[tile] - 1].bandwidth
                                elif quality_order == 'hl':
                                    optimized_quality_selections[(tile,fov_group_id)] = repr_list[len(self.resolution_ladder) - optimized_results[tile]].id
                                    self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - optimized_results[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - optimized_results[tile]].bandwidth}")
                                    self.previous_bitrates[tile] = repr_list[len(self.resolution_ladder) - optimized_results[tile]].bandwidth

                if mod_scheduler == "all_tiles" and solution_count == 0 :
                    infeasible_qualities = {i: 1 for i in range(0, n_tiles)}
                    self.previous_quality_selection = infeasible_qualities
                    for fov_group, tiles in fov_groups.items():
                        for tile in tiles:
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh':
                                optimized_quality_selections[(tile,fov_group)] = repr_list[0].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                            elif quality_order == 'hl':
                                optimized_quality_selections[(tile,fov_group)] = repr_list[-1].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")
                elif mod_scheduler == "fov_tiles" and solution_count == 0 :
                    for tile in left_tiles_selected:
                            self.previous_quality_selection[tile] = 1
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh':
                                optimized_quality_selections[(tile,fov_group_id)] = repr_list[0].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                                self.previous_bitrates[tile] = repr_list[0].bandwidth
                            elif quality_order == 'hl':
                                optimized_quality_selections[(tile,fov_group_id)] = repr_list[-1].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")
                                self.previous_bitrates[tile] = repr_list[0].bandwidth 



            
            pop_size = 30 
            dim = 3
            bounds = [(0.5, 1), (0.1, 0.8), (0.01, 0.2)]
            max_iter = 40

            def fitness_function(weights):
                return evalute_priority_weight(weights)
            
            best_solution, best_fitness = gwo(pop_size, dim, bounds, max_iter, fitness_function)
            print(f"Best solution: {best_solution}")
            print(f"Best fitness: {best_fitness}")

                

            # def initialize_population(population_size, num_weights, starting_point):
            #     population = [{'weights': np.array(starting_point), 'fitness': 0}]
            #     for _ in range(population_size - 1):
            #         weights = np.random.rand(num_weights)
            #         population.append({'weights': weights, 'fitness': 0})
            #         pdb.set_trace()
            #     return population
            
            
            # def fitness_function(weights, adaptation_sets, fov_groups, fov_group_id, scheduler_mode):
            #     self.priority_weight[0], self.priority_weight[1], self.priority_weight[2] = weights

            #     FoV = fov_groups[1]
            #     near_FoV = fov_groups[2]
            #     mod_scheduler = scheduler_mode #inputs of scheduler and update_selection
            #     left_tiles, left_tiles_selected = left_tiles_selector(fov_groups, fov_group_id)

            #     n_tiles = number_of_tiles(adaptation_sets)
            #     n_res = len(self.resolution_ladder)
            #     resolutions = self.resolution_ladder
            #     bitrate_ladder = bitrate_ladder_constructor(adaptation_sets, left_tiles)
            #     bw_penalty_factor = compute_bw_penalty()
            #     seg_duration = 1

            #     previous_qualities = {i: 1 for i in range(0, n_tiles)}
            #     previous_bitrates = {}
            #     if mod_scheduler == "all_tiles":
            #         if self.previous_quality_selection:
            #             previous_qualities = self.previous_quality_selection
            #         total_previous_bitrates = 0
            #         seg_duration = 1
            #     elif mod_scheduler == "fov_tiles":
            #         if not self.previous_quality_selection:
            #             self.previous_quality_selection = {i: 1 for i in range(0, n_tiles)}
            #         else:
            #             if fov_group_id == 1:
            #                 previous_qualities = self.previous_quality_selection
            #                 previous_bitrates = self.previous_bitrates
            #                 seg_duration = 1
            #             else:
            #                 left_tiles_prev, left_tiles_prev_selected = left_tiles_selector(fov_groups, fov_group_id - 1)
            #                 for tile in left_tiles_prev_selected:
            #                     previous_qualities[tile] = self.previous_quality_selection[tile]
            #                     previous_bitrates[tile] = self.previous_bitrates[tile]
            #                 seg_duration = 0.33
            #         total_previous_bitrates = sum(previous_bitrates.values())

            #     previous_qualities = {key: resolutions[value - 1] for key, value in previous_qualities.items()}

            #     # --------- Model -----------
            #     model = pyo.ConcreteModel()
            #     if mod_scheduler == "all_tiles":
            #         model.N = pyo.RangeSet(0, n_tiles - 1)
            #         model.N_quality = pyo.RangeSet(0, n_tiles - 1)
            #     elif mod_scheduler == "fov_tiles":
            #         model.N = pyo.Set(initialize=left_tiles)
            #         model.N_quality = pyo.RangeSet(0, n_tiles - 1)
            #     model.M = pyo.RangeSet(1, n_res)
            #     if self.bandwidth_meter.bandwidth <= self.bw_limit * self.bw_error_tolerance:
            #         model.BW = self.bandwidth_meter.bandwidth
            #     else:
            #         model.BW = self.bw_limit * 1.1
            #     model.buffer_level = self.buffer_manager.buffer_level

            #     model.RES = pyo.Param(model.M, initialize={i + 1: res for i, res in enumerate(resolutions)})
            #     model.prev_qualities = pyo.Param(model.N_quality, initialize=previous_qualities)
            #     model.bitrate = pyo.Param(model.N, model.M, initialize=bitrate_ladder)
            #     model.RES_th = pyo.Param(model.N, initialize=res_thereshold_selector(fov_groups, left_tiles))
            #     model.avg_throughput = pyo.Param(model.N, initialize=caclulate_average_throughput(self.analyzer.get_tile_segment_throughput(), left_tiles))
            #     model.avg_dl_time = pyo.Param(model.N, initialize=caclulate_average_throughput(self.analyzer.get_tile_segment_dl_time(), left_tiles))
            #     model.X = pyo.Var(model.N, model.M, within=pyo.Boolean)

            #     def obj_rule(model):
            #         return (sum(
            #             model.X[i, j] * model.RES[j] * (
            #                 self.priority_weight[0] if i in FoV else self.priority_weight[1] if i in near_FoV else self.priority_weight[2]
            #             )
            #             for j in model.M for i in model.N)
            #         )
            #     model.obj = pyo.Objective(sense=pyo.maximize, rule=obj_rule)

            #     def const1(model, i):
            #         return sum(model.X[i, j] for j in model.M) == 1
            #     model.const1 = pyo.Constraint(model.N, rule=const1)

            #     def const2(model, i):
            #         if i in FoV:
            #             return sum(model.RES[j] * model.X[i, j] for j in model.M) <= model.RES_th[i]
            #         elif i in near_FoV:
            #             return sum(model.RES[j] * model.X[i, j] for j in model.M) <= model.RES_th[i]
            #         else:
            #             return sum(model.RES[j] * model.X[i, j] for j in model.M) <= model.RES_th[i]
            #     model.const2 = pyo.Constraint(model.N, rule=const2)

            #     def const3(model, i):
            #         if i in FoV:
            #             return model.prev_qualities[i] <= sum(model.RES[j] * model.X[i, j] for j in model.M)
            #         else:
            #             return pyo.Constraint.Skip
            #     model.const3 = pyo.Constraint(model.N, rule=const3)

            #     def const4(model):
            #         return sum(model.X[i, j] * model.bitrate[i, j] for j in model.M for i in model.N) <= (
            #                 bw_penalty_factor * model.BW) - total_previous_bitrates
            #     model.const4 = pyo.Constraint(rule=const4)

            #     def const5(model):
            #         return self.panic_buffer <= model.buffer_level - sum(
            #             model.X[i, j] * model.avg_dl_time[i] for j in model.M for i in model.N) + seg_duration
            #     model.const5 = pyo.Constraint(rule=const5)

            #     solver = pyo.SolverFactory('gurobi')
            #     results = solver.solve(model, tee=False)

            #     if results.solver.termination_condition == TerminationCondition.infeasible:
            #         return 0
            #     elif results.solver.termination_condition == TerminationCondition.optimal:
            #         optimized_results = optimized_result_output(model, results, solver, left_tiles)
            #         return sum(optimized_results.values())
                
            #     def select_mating_pool(population, fitness, num_parents):
            #         parents = np.empty((num_parents, population.shape[1]))
            #         for parent_num in range(num_parents):
            #             max_fitness_idx = np.where(fitness == np.max(fitness))
            #             parents[parent_num, :] = population[max_fitness_idx[0][0], :]
            #             fitness[max_fitness_idx[0][0]] = -999999
            #         return parents
            #     def crossover(parents, offspring_size):
            #         offspring = np.empty(offspring_size)
            #         crossover_point = np.uint8(offspring_size[1] / 2)
            #         for k in range(offspring_size[0]):
            #             parent1_idx = k % parents.shape[0]
            #             parent2_idx = (k + 1) % parents.shape[0]
            #             offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            #             offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
            #         return offspring
            #     def mutation(offspring_crossover):
            #         for idx in range(offspring_crossover.shape[0]):
            #             random_value = np.random.uniform(-1.0, 1.0, 1)
            #             offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
            #         return offspring_crossover
                
            #     def ga_optimizer(population_size, num_weights, num_generations):
            #         population = initialize_population(population_size, num_weights)
            #         for generation in range(num_generations):
            #             fitness = np.empty(population_size)
            #             for i in range(population_size):
            #                 fitness[i] = fitness_function(population[i], adaptation_sets, fov_groups, fov_group_id, scheduler_mode)
            #             parents = select_mating_pool(population, fitness, num_parents)
            #             offspring_crossover = crossover(parents, offspring_size=(population_size[0] - parents.shape[0], num_weights))
            #             offspring_mutation = mutation(offspring_crossover)
            #             population[0:parents.shape[0], :] = parents
            #             population[parents.shape[0]:, :] = offspring_mutation
            #         fitness = np.empty(population_size)
            #         for i in range(population_size):
            #             fitness[i] = fitness_function(population[i], adaptation_sets, fov_groups, fov_group_id, scheduler_mode)
            #         max_fitness_idx = np.where(fitness == np.max(fitness))
            #         return population[max_fitness_idx[0][0], :]
                
            

       
        def quality_selection(adaptation_sets: AdaptationSet, fov_groups: Dict[int, list], fov_group_id:int, scheduler_mode:str) -> Dict[int, int]:
            """
            This function selects the quality for a specific FoV group.

            Parameters:
            adaptation_set (AdaptationSet): The adaptation sets.
            fov_groups: The different FoV groups and their assigned tiles.

            Returns:
            int: quality (representation) for each FoV group.
            """
            quality_selection = {}
            if scheduler_mode == "all_tiles":
                for fov_group, tiles in fov_groups.items():
                    if fov_group == 1:
                        for tile in tiles:
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh':
                                quality_selection[(tile,fov_group)] = repr_list[-1].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")
                            elif quality_order == 'hl':
                                quality_selection[(tile,fov_group)] = repr_list[0].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                    elif fov_group == 2:
                        for tile in tiles:
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh':
                                quality_selection[(tile,fov_group)] = repr_list[0].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                            elif quality_order == 'hl':
                                quality_selection[(tile,fov_group)] = repr_list[-1].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")
                    elif fov_group == 3:
                        for tile in tiles:
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh':
                                quality_selection[(tile,fov_group)] = repr_list[0].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                            elif quality_order == 'hl':
                                quality_selection[(tile,fov_group)] = repr_list[-1].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")
                            
            if scheduler_mode == "fov_tiles":
                for fov_group, tiles in fov_groups.items():
                    if fov_group == fov_group_id:
                        for tile in tiles:
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if fov_group_id == 1:
                                if quality_order == 'lh':
                                    quality_selection[(tile,fov_group)] = repr_list[-1].id
                                    self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")
                                elif quality_order == 'hl':
                                    quality_selection[(tile,fov_group)] = repr_list[0].id
                                    self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                            else:
                                if quality_order == 'lh':
                                    quality_selection[(tile,fov_group)] = repr_list[0].id
                                    self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                                elif quality_order == 'hl':
                                    quality_selection[(tile,fov_group)] = repr_list[-1].id
                                    self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")
            
            
            return quality_selection
    
        def quality_selection_ml(adaptation_sets: AdaptationSet, fov_groups: Dict[int, list], fov_group_id:int, scheduler_mode:str) -> Dict[int, int]:
            """
            This function selects the quality for a specific FoV group using a Machine Learning model.

            Parameters:
            adaptation_set (AdaptationSet): The adaptation sets.
            fov_groups: The different FoV groups and their assigned tiles.

            Returns:
            int: quality (representation) for each FoV group.
            """

            quality_selection_ml = {}
            ml_results_dict = {}
            left_tiles, left_tiles_selected = left_tiles_selector(fov_groups, fov_group_id)
            bw = self.bandwidth_meter.bandwidth
            buffer_level = self.buffer_manager.buffer_level
            model = jl.load('xgb_model.pkl')
            # finding stalls
            stall_counter = self.analyzer.get_stalls_number()
            if self.prev_stall_counter < stall_counter:
                self.log.info(f"Stalls are increasing: {stall_counter}")
                stall_flag = 1
                self.prev_stall_counter = stall_counter
            else: 
                stall_flag = 0
                self.prev_stall_counter = stall_counter
                self.log.info(f"no stall: {stall_counter}")


            for fov_group, tiles in fov_groups.items():
                for tile in tiles:
                    new_data = pd.DataFrame({
                        'index': [index],
                        'adap_set_id': [tile],
                        'buffer_level': [buffer_level],
                        'adaptation_throughput': [bw],
                        'fov_group_id': [fov_group],
                        'stall': [stall_flag]
                    })
                    ml_results_dict[tile] = model.predict(new_data).item() + 1
                    


            # self.log.info(f"ML Results are: {ml_results_dict}")
            if scheduler_mode == "all_tiles":
                for fov_group, tiles in fov_groups.items():
                    for tile in tiles:
                        adaptation_set = adaptation_sets[tile]
                        repr_list = qualities(adaptation_set)
                        quality_order = quality_order_selector(repr_list)
                        if quality_order == 'lh':
                            quality_selection_ml[(tile,fov_group)] = repr_list[ml_results_dict[tile] - 1].id
                            self.log.info(f"Tile {tile} with resolution {repr_list[ml_results_dict[tile] - 1].height} and bitrate {repr_list[ml_results_dict[tile] - 1].bandwidth}")
                        elif quality_order == 'hl':
                            quality_selection_ml[(tile,fov_group)] = repr_list[len(self.resolution_ladder) - ml_results_dict[tile]].id
                            self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - ml_results_dict[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - ml_results_dict[tile]].bandwidth}")
            
            elif scheduler_mode == "fov_tiles":
                for fov_group, tiles in fov_groups.items():
                    if fov_group == fov_group_id:
                        for tile in tiles:
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh':
                                quality_selection_ml[(tile,fov_group)] = repr_list[-1].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")
                            elif quality_order == 'hl':
                                quality_selection_ml[(tile,fov_group)] = repr_list[0].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
            
            return quality_selection_ml

        def quality_selection_equal(adaptation_sets: AdaptationSet, fov_groups: Dict[int, list], fov_group_id:int, scheduler_mode:str) -> Dict[int, int]:
            """
            This function equally distribute the quality among all tiles according to the available BW.

            Parameters:
            adaptation_set (AdaptationSet): The adaptation sets.
            fov_groups: The different FoV groups and their assigned tiles.

            Returns:
            int: quality (representation) for each FoV group.
            """

            quality_selection_equal = {}
            bw = self.bandwidth_meter.bandwidth
            n_tiles = number_of_tiles(adaptation_sets)
            allocated_bitrate = None
            allocated_bitrate_index = None
            left_tiles, left_tiles_selected = left_tiles_selector(fov_groups, fov_group_id)

            if bw <= self.bw_limit * self.bw_error_tolerance: 
                bw = bw 
            else: 
                bw = self.bw_limit * 1

            if scheduler_mode == "all_tiles":
                for index, item in enumerate(self.bitrate_ladder):
                    if item * n_tiles <= bw:
                        if allocated_bitrate is None or item > allocated_bitrate:
                            allocated_bitrate = item
                            allocated_bitrate_index = index
                    if allocated_bitrate is None:
                        self.log.info(f"The available bandwidth is not enough to support all tiles with the lowest quality - bw is {bw}")
                        self.not_enough_bw_counter += 1
                        allocated_bitrate = self.bitrate_ladder[0]
                        allocated_bitrate_index = 0
                self.log.info(f"Allocated Bitrate/Resolution is: {allocated_bitrate}/{self.resolution_ladder[allocated_bitrate_index]}")

                for fov_group, tiles in fov_groups.items():
                    for tile in tiles:
                        adaptation_set = adaptation_sets[tile]
                        repr_list = qualities(adaptation_set)
                        quality_order = quality_order_selector(repr_list)
                        
                        if quality_order == 'lh':
                            quality_selection_equal[(tile,fov_group)] = repr_list[allocated_bitrate_index].id
                            self.log.info(f"Tile {tile} with resolution {repr_list[allocated_bitrate_index].height} and bitrate {repr_list[allocated_bitrate_index].bandwidth}")
                        elif quality_order == 'hl':
                            quality_selection_equal[(tile,fov_group)] = repr_list[len(self.bitrate_ladder) - allocated_bitrate_index - 1].id
                            self.log.info(f"Tile {tile} with resolution {repr_list[len(self.bitrate_ladder) - allocated_bitrate_index - 1].height} and bitrate {repr_list[len(self.bitrate_ladder) - allocated_bitrate_index - 1].bandwidth}")
            
            elif scheduler_mode == "fov_tiles":
                for index, item in enumerate(self.bitrate_ladder):
                    if item * len(left_tiles_selected) <= bw:
                        if allocated_bitrate is None or item > allocated_bitrate:
                            allocated_bitrate = item
                            allocated_bitrate_index = index
                self.log.info(f"Allocated Bitrate/Resolution is: {allocated_bitrate}/{self.resolution_ladder[allocated_bitrate_index]}")

                for fov_group, tiles in fov_groups.items():
                    if fov_group == fov_group_id:
                        for tile in tiles:
                            # pdb.set_trace()
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh':
                                quality_selection_equal[(tile,fov_group)] = repr_list[allocated_bitrate_index].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[allocated_bitrate_index].height} and bitrate {repr_list[allocated_bitrate_index].bandwidth}")
                            elif quality_order == 'hl':
                                quality_selection_equal[(tile,fov_group)] = repr_list[len(self.bitrate_ladder) - allocated_bitrate_index - 1].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[len(self.bitrate_ladder) - allocated_bitrate_index - 1].height} and bitrate {repr_list[len(self.bitrate_ladder) - allocated_bitrate_index - 1].bandwidth}")

            return quality_selection_equal
        
        def quality_selection_full_advanced(adaptation_sets: AdaptationSet, fov_groups: Dict[int, list], fov_group_id:int, scheduler_mode:str) -> Dict[int, int]:
            """
            This function selects the quality for a specific FoV group based on Full Advanced Algorithm

            Parameters:
            adaptation_set (AdaptationSet): The adaptation sets.
            fov_groups: The different FoV groups and their assigned tiles.

            Returns:
            int: quality (representation) for each FoV group.
            """


            quality_selection_full_advanced = {}
            tile_res = {}
            best_combination = None
            bw = self.bandwidth_meter.bandwidth
            n_tiles = number_of_tiles(adaptation_sets)
            left_tiles, left_tiles_selected = left_tiles_selector(fov_groups, fov_group_id)
            bitrate_ladder = bitrate_ladder_constructor(adaptation_sets, left_tiles)
            if bw <= self.bw_limit * self.bw_error_tolerance:
                bw = bw
            else:
                bw = self.bw_limit * 1

            def sum_allocated_bitrate (bitrate_ladder, selected_tiles_resolutions):
                sum_bitrate = 0
                for tile, res in selected_tiles_resolutions.items():
                    sum_bitrate += bitrate_ladder[(tile, res)]
                return sum_bitrate

            def res_tile_selector (fov_groups, res_th_index):
                for fov_group, tiles in fov_groups.items():
                    for tile in tiles:
                        if fov_group == 1: 
                            tile_res[tile] = res_th_index[0] 
                        elif fov_group == 2:
                            tile_res[tile] = res_th_index[1] 
                        elif fov_group == 3:
                            tile_res[tile] = res_th_index[2]
                return tile_res


            for zone1_index in range(len(self.resolution_ladder), 0, -1): #bitrate ladder starts from low to high (index resolution starts from 1)
                for zone2_index in range(len(self.resolution_ladder), 0, -1): 
                    for zone3_index in range(1, 2, 1): #least quality to zone 3
                        tile_res = res_tile_selector(fov_groups, [zone1_index,zone2_index,zone3_index]) #degarde the quality of each zone step by step
                        if sum_allocated_bitrate(bitrate_ladder, tile_res) <= bw:
                            best_combination = [zone1_index, zone2_index, zone3_index]
                            max_sum = sum_allocated_bitrate(bitrate_ladder, tile_res)
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

            if best_combination is None:
                self.log.info(f"The available bandwidth is not enough to support all tiles with the lowest quality - bw is {bw}")
                self.not_enough_bw_counter += 1
                max_sum = 0
                tile_res = {i: 1 for i in range(0, n_tiles)}

            self.log.info(f"Best combination of indexes: {best_combination}")
            self.log.info(f"Maximum sum of allocated bitrates (in bandwidth) {max_sum}({bw})")
            if best_combination is not None:
                tile_res = res_tile_selector(fov_groups, best_combination)

            
            for fov_group, tiles in fov_groups.items():
                for tile in tiles:
                    adaptation_set = adaptation_sets[tile]
                    repr_list = qualities(adaptation_set)
                    quality_order = quality_order_selector(repr_list)
                    if quality_order == 'lh':
                        quality_selection_full_advanced[(tile,fov_group)] = repr_list[tile_res[tile]].id
                        self.log.info(f"Tile {tile} with resolution {repr_list[tile_res[tile]].height} and bitrate {repr_list[tile_res[tile]].bandwidth}")
                    elif quality_order == 'hl':
                        quality_selection_full_advanced[(tile,fov_group)] = repr_list[len(self.resolution_ladder) - tile_res[tile]].id
                        self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - tile_res[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - tile_res[tile]].bandwidth}")

            return quality_selection_full_advanced


        def quality_selection_buffer(adaptation_sets: AdaptationSet, fov_groups: Dict[int, list], fov_group_id:int, scheduler_mode:str) -> Dict[int, int]:
            """
            This function tries to assign qualties based on available bw and buffer health.

            Parameters:
            adaptation_set (AdaptationSet): The adaptation sets.
            fov_groups: The different FoV groups and their assigned tiles.

            Returns:
            int: quality (representation) for each FoV group.
            """

            quality_selection_buffer = {}
            tile_res = {}
            best_combination = None
            bw = self.bandwidth_meter.bandwidth
            buffer_level = self.buffer_manager.buffer_level
            n_tiles = number_of_tiles(adaptation_sets)
            left_tiles, left_tiles_selected = left_tiles_selector(fov_groups, fov_group_id)
            bitrate_ladder = bitrate_ladder_constructor(adaptation_sets, left_tiles) #the order of the bitrate ladder is from low to high (tile starts from 0 and resolution starts from 1)

            if bw <= self.bw_limit * self.bw_error_tolerance:
                bw = bw
            else:
                bw = self.bw_limit * 1

            def sum_allocated_bitrate (bitrate_ladder, selected_tiles_resolutions):
                sum_bitrate = 0
                for tile, res in selected_tiles_resolutions.items():
                    sum_bitrate += bitrate_ladder[(tile, res)]
                return sum_bitrate

            def res_tile_selector (fov_groups, res_th_index):
                for fov_group, tiles in fov_groups.items():
                    for tile in tiles:
                        if fov_group == 1: 
                            tile_res[tile] = res_th_index[0] 
                        elif fov_group == 2:
                            tile_res[tile] = res_th_index[1] 
                        elif fov_group == 3:
                            tile_res[tile] = res_th_index[2]
                return tile_res

            if scheduler_mode == "all_tiles":
                if buffer_level < self.panic_buffer:
                    for fov_group, tiles in fov_groups.items():
                        for tile in tiles:
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh':
                                quality_selection_buffer[(tile,fov_group)] = repr_list[0].id #lowest quality
                                self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                            elif quality_order == 'hl':
                                quality_selection_buffer[(tile,fov_group)] = repr_list[-1].id 
                                self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")

                else: #buffer is healthy
                    for zone1_index in range(len(self.resolution_ladder), 0, -1): #bitrate ladder starts from low to high (index resolution starts from 1)
                        for zone2_index in range(len(self.resolution_ladder), 0, -1): 
                            for zone3_index in range(1, 2, 1): #least quality to zone 3 
                                tile_res = res_tile_selector(fov_groups, [zone1_index,zone2_index,zone3_index]) #degarde the quality of each zone step by step
                                if sum_allocated_bitrate(bitrate_ladder, tile_res) <= bw:
                                    best_combination = [zone1_index, zone2_index, zone3_index]
                                    max_sum = sum_allocated_bitrate(bitrate_ladder, tile_res)
                                    break
                            else:
                                continue
                            break
                        else:
                            continue
                        break

                    if best_combination is None:
                        self.log.info(f"The available bandwidth is not enough to support all tiles with the lowest quality - bw is {bw}")
                        self.not_enough_bw_counter += 1
                        max_sum = 0
                        tile_res = {i: 1 for i in range(0, n_tiles)}

                    self.log.info(f"Best combination of indexes: {best_combination}")
                    self.log.info(f"Maximum sum of allocated bitrates (in bandwidth) {max_sum}({bw})")
                    if best_combination is not None:
                        tile_res = res_tile_selector(fov_groups, best_combination)

                    
                    for fov_group, tiles in fov_groups.items():
                        for tile in tiles:
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh':
                                quality_selection_buffer[(tile,fov_group)] = repr_list[tile_res[tile]].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[tile_res[tile]].height} and bitrate {repr_list[tile_res[tile]].bandwidth}")
                            elif quality_order == 'hl':
                                quality_selection_buffer[(tile,fov_group)] = repr_list[len(self.resolution_ladder) - tile_res[tile]].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - tile_res[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - tile_res[tile]].bandwidth}")

            return quality_selection_buffer


        def quality_selection_greedy(adaptation_sets: AdaptationSet, fov_groups: Dict[int, list], fov_group_id:int, scheduler_mode:str) -> Dict[int, int]:
            """
            This function tries to assign qualties based on available bw and buffer health.

            Parameters:
            adaptation_set (AdaptationSet): The adaptation sets.
            fov_groups: The different FoV groups and their assigned tiles.

            Returns:
            int: quality (representation) for each FoV group.
            """

            quality_selection_greedy = {}
            tile_res = {}
            best_combination = None
            previous_bitrates = {}
            total_previous_bitrates = 0
            bw = self.bandwidth_meter.bandwidth
            buffer_level = self.buffer_manager.buffer_level
            n_tiles = number_of_tiles(adaptation_sets)
            left_tiles, left_tiles_selected = left_tiles_selector(fov_groups, fov_group_id)
            bitrate_ladder = bitrate_ladder_constructor(adaptation_sets, left_tiles) #the order of the bitrate ladder is from low to high (tile starts from 0 and resolution starts from 1)
            resolution_sorted = sorted(self.resolution_ladder, reverse=True) #from high to low

            if bw <= self.bw_limit * self.bw_error_tolerance:
                bw = bw
            else:
                bw = self.bw_limit * 1

            def sum_allocated_bitrate (bitrate_ladder, selected_tiles_resolutions):
                sum_bitrate = 0
                for tile, res in selected_tiles_resolutions.items():
                    sum_bitrate += bitrate_ladder[(tile, res)]
                return sum_bitrate

            def res_tile_selector (fov_groups, res_th_index):
                for fov_group, tiles in fov_groups.items():
                    for tile in tiles:
                        if fov_group == 1: 
                            tile_res[tile] = res_th_index[0] 
                        elif fov_group == 2:
                            tile_res[tile] = res_th_index[1] 
                        elif fov_group == 3:
                            tile_res[tile] = res_th_index[2]
                return tile_res
            
            def res_tile_selector_fov (fov_groups, res_th_index, left_tiles):
                for fov_group, tiles in fov_groups.items():
                    for tile in tiles:
                        if fov_group == 1 and tile in left_tiles:
                            tile_res[tile] = res_th_index[0] 
                        elif fov_group == 2 and tile in left_tiles:
                            tile_res[tile] = res_th_index[1] 
                        elif fov_group == 3 and tile in left_tiles:
                            tile_res[tile] = res_th_index[2]
                return tile_res

            if scheduler_mode == "all_tiles":
                if buffer_level < self.panic_buffer:
                    for fov_group, tiles in fov_groups.items():
                        for tile in tiles:
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh':
                                quality_selection_greedy[(tile,fov_group)] = repr_list[0].id #lowest quality
                                self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                            elif quality_order == 'hl':
                                quality_selection_greedy[(tile,fov_group)] = repr_list[-1].id 
                                self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")

                else: #buffer is healthy
                    for zone1_index in range(len(self.resolution_ladder), 0, -1): #bitrate ladder starts from low to high (index resolution starts from 1)
                        for zone2_index in range(len(self.resolution_ladder), 0, -1): 
                            for zone3_index in range(len(self.resolution_ladder), 0, -1): 
                                tile_res = res_tile_selector(fov_groups, [zone1_index,zone2_index,zone3_index]) #degarde the quality of each zone step by step
                                if sum_allocated_bitrate(bitrate_ladder, tile_res) <= bw:
                                    best_combination = [zone1_index, zone2_index, zone3_index]
                                    max_sum = sum_allocated_bitrate(bitrate_ladder, tile_res)
                                    break
                            else:
                                continue
                            break
                        else:
                            continue
                        break

                    if best_combination is None:
                        self.log.info(f"The available bandwidth is not enough to support all tiles with the lowest quality - bw is {bw}")
                        self.not_enough_bw_counter += 1
                        max_sum = 0
                        tile_res = {i: 1 for i in range(0, n_tiles)}

                    self.log.info(f"Best combination of indexes: {best_combination}")
                    self.log.info(f"Maximum sum of allocated bitrates (in bandwidth) {max_sum}({bw})")
                    if best_combination is not None:
                        tile_res = res_tile_selector(fov_groups, best_combination)

                    
                    for fov_group, tiles in fov_groups.items():
                        for tile in tiles:
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh':
                                quality_selection_greedy[(tile,fov_group)] = repr_list[tile_res[tile]].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[tile_res[tile]].height} and bitrate {repr_list[tile_res[tile]].bandwidth}")
                            elif quality_order == 'hl':
                                quality_selection_greedy[(tile,fov_group)] = repr_list[len(self.resolution_ladder) - tile_res[tile]].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - tile_res[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - tile_res[tile]].bandwidth}")
            
            
            elif scheduler_mode == "fov_tiles":
                if buffer_level < self.panic_buffer:
                    for fov_group, tiles in fov_groups.items():
                        if fov_group == fov_group_id:
                            for tile in tiles:
                                adaptation_set = adaptation_sets[tile]
                                repr_list = qualities(adaptation_set)
                                quality_order = quality_order_selector(repr_list)
                                if quality_order == 'lh':
                                    quality_selection_greedy[(tile,fov_group)] = repr_list[0].id
                                    self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                                elif quality_order == 'hl':
                                    quality_selection_greedy[(tile,fov_group)] = repr_list[-1].id
                                    self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")

                else:
                    if fov_group_id == 1:
                        previous_bitrates = self.previous_bitrates
                    else:
                        left_tiles_prev, left_tiles_prev_selected = left_tiles_selector(fov_groups, fov_group_id -1)
                        for tile in left_tiles_prev_selected:
                            previous_bitrates[tile] = self.previous_bitrates[tile]
                        
                    total_previous_bitrates = sum(previous_bitrates.values())
                    print(f"Total Previous Bitrates: {total_previous_bitrates} - Previous Bitrates: {previous_bitrates}")

                    if fov_group_id == 1:
                        for zone1_index in range(len(self.resolution_ladder), 0, -1):
                            for zone2_index in range(len(self.resolution_ladder), 0, -1):
                                for zone3_index in range(len(self.resolution_ladder), 0, -1):
                                    tile_res = res_tile_selector_fov(fov_groups, [zone1_index,zone2_index,zone3_index], left_tiles)
                                    if sum_allocated_bitrate(bitrate_ladder, tile_res) <= bw:
                                        best_combination = [zone1_index, zone2_index, zone3_index]
                                        max_sum = sum_allocated_bitrate(bitrate_ladder, tile_res)
                                        break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                    elif fov_group_id == 2:
                        for zone2_index in range(len(self.resolution_ladder), 0, -1):
                            for zone3_index in range(len(self.resolution_ladder), 0, -1):
                                tile_res = res_tile_selector_fov(fov_groups, [len(self.resolution_ladder), zone2_index,zone3_index], left_tiles)
                                if sum_allocated_bitrate(bitrate_ladder, tile_res) <= bw - total_previous_bitrates:
                                    best_combination = [len(self.resolution_ladder), zone2_index, zone3_index]
                                    max_sum = sum_allocated_bitrate(bitrate_ladder, tile_res)
                                    break
                            else:
                                continue
                            break
                    elif fov_group_id == 3:
                        for zone3_index in range(len(self.resolution_ladder), 0, -1):
                            tile_res = res_tile_selector_fov(fov_groups, [len(self.resolution_ladder), len(self.resolution_ladder), zone3_index], left_tiles)
                            if sum_allocated_bitrate(bitrate_ladder, tile_res) <= bw - total_previous_bitrates:
                                best_combination = [len(self.resolution_ladder), len(self.resolution_ladder), zone3_index]
                                max_sum = sum_allocated_bitrate(bitrate_ladder, tile_res)
                                break
                            else:
                                continue

                    if best_combination is None:
                        self.log.info(f"The available bandwidth is not enough to support all tiles with the lowest quality - bw is {bw}")
                        max_sum = 0
                        self.not_enough_bw_counter += 1
                        tile_res = {i: 1 for i in left_tiles_selected}

                    self.log.info(f"Best combination of indexes: {best_combination}")
                    self.log.info(f"Maximum sum of allocated bitrates (in bandwidth) {max_sum}({bw})")
                    if best_combination is not None:
                        tile_res = res_tile_selector_fov(fov_groups, best_combination, left_tiles)

                    for fov_group, tiles in fov_groups.items():
                        if fov_group == fov_group_id:
                            for tile in tiles:
                                adaptation_set = adaptation_sets[tile]
                                repr_list = qualities(adaptation_set)
                                quality_order = quality_order_selector(repr_list)
                                if quality_order == 'lh':
                                    quality_selection_greedy[(tile,fov_group)] = repr_list[tile_res[tile]].id
                                    self.log.info(f"Tile {tile} with resolution {repr_list[tile_res[tile]].height} and bitrate {repr_list[tile_res[tile]].bandwidth}")
                                    self.previous_bitrates[tile] = repr_list[tile_res[tile]].bandwidth
                                elif quality_order == 'hl':
                                    quality_selection_greedy[(tile,fov_group)] = repr_list[len(self.resolution_ladder) - tile_res[tile]].id
                                    self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - tile_res[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - tile_res[tile]].bandwidth}")
                                    self.previous_bitrates[tile] = repr_list[len(self.resolution_ladder) - tile_res[tile]].bandwidth


            if fov_group_id == 3:
                self.previous_bitrates = {}
                
            return quality_selection_greedy

        def quality_selection_tbra(adaptation_sets: AdaptationSet, fov_groups: Dict[int, list], fov_group_id:int, scheduler_mode:str) -> Dict[int, int]:
            """
            This function selects the quality for a specific FoV group using a TBRA model.

            Parameters:
            adaptation_set (AdaptationSet): The adaptation sets.
            fov_groups: The different FoV groups and their assigned tiles.

            Returns:
            int: quality (representation) for each FoV group.
            """
            

            mod_scheduler = scheduler_mode
            FoV = fov_groups[1]
            near_FoV = fov_groups[2]
            n_tiles = number_of_tiles(adaptation_sets)
            resolutions = self.resolution_ladder
            left_tiles, left_tiles_selected = left_tiles_selector(fov_groups, fov_group_id)
            bitrate_ladder = bitrate_ladder_constructor(adaptation_sets, left_tiles)


            previous_qualities = {i: 1 for i in range(0, n_tiles)}
            if mod_scheduler == "all_tiles":
                if self.previous_quality_selection:
                    previous_qualities = self.previous_quality_selection
            elif mod_scheduler == "fov_tiles":
                if not self.previous_quality_selection:
                    self.previous_quality_selection = {i: 1 for i in range(0, n_tiles)}
                else:
                    if fov_group_id == 1:
                        previous_qualities = self.previous_quality_selection
                    else:
                        left_tiles_prev, left_tiles_prev_selected = left_tiles_selector(fov_groups, fov_group_id -1)
                        for tile in left_tiles_prev_selected:
                            previous_qualities[tile] = self.previous_quality_selection[tile]
                

            previous_qualities = {key: resolutions[value - 1] for key, value in previous_qualities.items()}
            

            
            model = pyo.ConcreteModel()

            #sets
            if mod_scheduler == "all_tiles":
                model.N = pyo.RangeSet(0, n_tiles - 1) #tile number starts from 0
                model.N_quality = pyo.RangeSet(0, n_tiles - 1)
            elif mod_scheduler == "fov_tiles":
                model.N = pyo.Set(initialize = left_tiles)
                model.N_quality = pyo.RangeSet(0, n_tiles - 1) 
            model.M = pyo.RangeSet(1, len(self.resolution_ladder))
            
            #parameters
            model.RES = pyo.Param(model.M, initialize = {i+1: res for i, res in enumerate(resolutions)}) #default order: low - high
            model.bitrate = pyo.Param(model.N, model.M, initialize=bitrate_ladder)
            if self.bandwidth_meter.bandwidth <= self.bw_limit * self.bw_error_tolerance:
                model.BW = self.bandwidth_meter.bandwidth
            else:
                model.BW = self.bw_limit * 1
            model.buffer_level = self.buffer_manager.buffer_level
            model.prev_qualities = pyo.Param(model.N_quality, initialize=previous_qualities)
            model.avg_dl_time = pyo.Param(model.N, initialize = caclulate_average_throughput(self.analyzer.get_tile_segment_dl_time(), left_tiles))

            #variables
            model.X = pyo.Var(model.N, model.M, within=pyo.Boolean)
            model.T_i = pyo.Var(within=pyo.NonNegativeReals)
            model.G_i = pyo.Var(model.N, within=pyo.NonNegativeReals)


            #objective function
            def obj_rule(model):
                Q_i = sum(model.X[i, j] * model.RES[j] * (
                        self.priority_weight_tbra[0] if i in FoV else self.priority_weight_tbra[1] if i in near_FoV else self.priority_weight_tbra[2]
                    )  for j in model.M for i in model.N) / len(model.N)
                # T_i = max(sum(model.X[i,j] * model.avg_dl_time[i] for j in model.M for i in model.N) - model.buffer_level, 0) 
                # G_i = sum(abs(model.X[i, j] * model.RES[j] - model.prev_qualities[i]) for j in model.M for i in model.N)
                return self.weight_obj_func[0] * Q_i - self.weight_obj_func[1] * model.T_i - self.weight_obj_func[2] * sum(model.G_i[i] for i in model.N)
            
            model.obj = pyo.Objective(sense=pyo.maximize, rule=obj_rule)

            #constraints
            def const1(model, i):
                return sum(model.X[i, j] for j in model.M) == 1
            model.const1 = pyo.Constraint(model.N, rule=const1)

            def const2(model):
                return sum(model.X[i, j] * model.bitrate[i, j] for j in model.M for i in model.N) <= model.BW
            model.const2 = pyo.Constraint(rule=const2)

            def const3(model):
                return model.T_i >= sum(model.X[i,j] * model.avg_dl_time[i] for j in model.M for i in model.N) - model.buffer_level
            model.const3 = pyo.Constraint(rule=const3)

            def const4(model, i, j):
                return model.G_i[i] >= model.X[i, j] * model.RES[j] - model.prev_qualities[i]
            model.const4 = pyo.Constraint(model.N, model.M, rule=const4)

            def const5(model, i, j):
                return  model.G_i[i] >= model.prev_qualities[i] - model.X[i, j] * model.RES[j]
            model.const5 = pyo.Constraint(model.N, model.M, rule=const5)

            #solve the model
            # model.pprint()
            # pdb.set_trace()
            solver = pyo.SolverFactory('gurobi')
            results = solver.solve(model, tee=False)

            #get the results
            if results.solver.termination_condition == TerminationCondition.infeasible:
                self.log.info(f"The solution is infeasible. ---- Bandwidth is {model.BW}")
                self.infeasible_solutions += 1
                optimized_quality_selections = {}
                solution_count = 0
            elif results.solver.termination_condition == TerminationCondition.optimal:
                optimized_results = optimized_result_output(model, results, solver, left_tiles)
                self.log.info(f"Optimized Results are: {optimized_results}") #tile number and selected resolution index (starts from 1) (from self.resolution_ladder)
                optimized_quality_selections = {}
                solution_count = 1

            if mod_scheduler == "all_tiles" and solution_count > 0:
                    self.previous_quality_selection = optimized_results
                    for fov_group, tiles in fov_groups.items():  
                        for tile in tiles:
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh': #index - 1 because the index starts from 1
                                optimized_quality_selections[(tile,fov_group)] = repr_list[optimized_results[tile] - 1].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[optimized_results[tile] - 1].height} and bitrate {repr_list[optimized_results[tile] - 1].bandwidth}")
                            elif quality_order == 'hl':
                                optimized_quality_selections[(tile,fov_group)] = repr_list[len(self.resolution_ladder) - optimized_results[tile]].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - optimized_results[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - optimized_results[tile]].bandwidth}")
            
            elif mod_scheduler == "fov_tiles" and solution_count > 0:
                    for tile in left_tiles_selected:
                            self.previous_quality_selection[tile] = optimized_results[tile]
                            adaptation_set = adaptation_sets[tile]
                            repr_list = qualities(adaptation_set)
                            quality_order = quality_order_selector(repr_list)
                            if quality_order == 'lh': #index - 1 because the index starts from 1 in the model
                                optimized_quality_selections[(tile,fov_group_id)] = repr_list[optimized_results[tile] - 1].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[optimized_results[tile] - 1].height} and bitrate {repr_list[optimized_results[tile] - 1].bandwidth}")
                                self.previous_bitrates[tile] = repr_list[optimized_results[tile] - 1].bandwidth
                            elif quality_order == 'hl':
                                optimized_quality_selections[(tile,fov_group_id)] = repr_list[len(self.resolution_ladder) - optimized_results[tile]].id
                                self.log.info(f"Tile {tile} with resolution {repr_list[len(self.resolution_ladder) - optimized_results[tile]].height} and bitrate {repr_list[len(self.resolution_ladder) - optimized_results[tile]].bandwidth}")
                                self.previous_bitrates[tile] = repr_list[len(self.resolution_ladder) - optimized_results[tile]].bandwidth

            if mod_scheduler == "all_tiles" and solution_count == 0 :
                infeasible_qualities = {i: 1 for i in range(0, n_tiles)}
                self.previous_quality_selection = infeasible_qualities
                for fov_group, tiles in fov_groups.items():
                    for tile in tiles:
                        adaptation_set = adaptation_sets[tile]
                        repr_list = qualities(adaptation_set)
                        quality_order = quality_order_selector(repr_list)
                        if quality_order == 'lh':
                            optimized_quality_selections[(tile,fov_group)] = repr_list[0].id
                            self.log.info(f"Tile {tile} with resolution {repr_list[0].height} and bitrate {repr_list[0].bandwidth}")
                        elif quality_order == 'hl':
                            optimized_quality_selections[(tile,fov_group)] = repr_list[-1].id
                            self.log.info(f"Tile {tile} with resolution {repr_list[-1].height} and bitrate {repr_list[-1].bandwidth}")


            
            return optimized_quality_selections





                    







        
        if self.mode_selection == "heuristic":
            final_selections = quality_selection(adaptation_sets, fov_groups, fov_group_id, scheduler_mode)
        elif self.mode_selection == "optimized":
            final_selections = quality_selection_optimized(adaptation_sets, fov_groups) #optimized quality selection
            # final_selections = ga_weight_selector(adaptation_sets, fov_groups, fov_group_id, scheduler_mode)
        elif self.mode_selection == "equal":
            final_selections = quality_selection_equal(adaptation_sets, fov_groups, fov_group_id, scheduler_mode)
        elif self.mode_selection == "greedy":
            final_selections = quality_selection_greedy(adaptation_sets, fov_groups, fov_group_id, scheduler_mode)
        elif self.mode_selection == "ml":
            final_selections = quality_selection_ml(adaptation_sets, fov_groups, fov_group_id, scheduler_mode)
        elif self.mode_selection == "tbra":
            final_selections = quality_selection_tbra(adaptation_sets, fov_groups, fov_group_id, scheduler_mode)
        elif self.mode_selection == "full_advanced":
            final_selections = quality_selection_full_advanced(adaptation_sets, fov_groups, fov_group_id, scheduler_mode)
        elif self.mode_selection == "buffer":
            final_selections = quality_selection_buffer(adaptation_sets, fov_groups, fov_group_id, scheduler_mode)
        # self.log.info(f"Final Selections are: {final_selections}")
        self.log.info(f"number of infeasible solutions are: {self.infeasible_solutions}/{self.not_enough_bw_counter}")



        return final_selections
