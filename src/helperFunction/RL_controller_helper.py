#!/usr/bin/env python

# Authors: Jungpyo Lee
# Create: Sep. 29. 2025
# Description: RL controller helper for integrating 2D Haptic Search models into ROS-based haptic search system.
#              This module provides a bridge between the ResidualRLController and the ROS haptic search framework.

import os
import sys
import numpy as np
from typing import Tuple, Dict, Any, Optional
import rospy

# Add the 2D Haptic Search project to the path
haptic_search_path = "/home/edg/Desktop/Jungpyo/Github/2D_Haptic_Search"
if haptic_search_path not in sys.path:
    sys.path.append(haptic_search_path)

try:
    from haptic_search.controllers.residual_rl_controller import ResidualRLController
    from haptic_search.controllers.heuristic_controller import HeuristicController
    rl_available = True
except ImportError as e:
    rospy.logwarn(f"Could not import RL controllers: {e}")
    rl_available = False


class RLControllerHelper:
    """
    Helper class to integrate RL models from 2D Haptic Search into ROS-based haptic search system.
    Provides a bridge between the ResidualRLController and the existing hapticSearch2DHelp framework.
    """
    
    def __init__(self, 
                 num_chambers: int = 4,
                 model_path: Optional[str] = None,
                 heuristic_mode: str = "greedy",
                 heuristic_step_size: float = 0.5,
                 heuristic_delta_yaw: float = 1.0,
                 heuristic_damping: float = 0.9,
                 step_lateral: float = 1.0,
                 step_yaw: float = 3.0,
                 residual_scale_xy: float = 0.5,
                 residual_scale_yaw: float = 0.5,
                 use_yaw_residual: str = "auto",
                 max_vacuum_pressure: float = 3.1171e4,
                 deterministic: bool = True,
                 clip_final_by_step: bool = True):
        """
        Initialize the RL controller helper.
        
        Args:
            num_chambers: Number of suction cup chambers (3, 4, 5, or 6)
            model_path: Path to the RL model file. If None, will use heuristic only.
            heuristic_mode: Heuristic controller mode ("greedy", "momentum")
            heuristic_step_size: Step size for heuristic controller
            heuristic_delta_yaw: Yaw step size for heuristic controller
            heuristic_damping: Damping factor for momentum controller
            step_lateral: Maximum lateral step size
            step_yaw: Maximum yaw step size
            residual_scale_xy: Scaling factor for residual actions in XY
            residual_scale_yaw: Scaling factor for residual actions in yaw
            use_yaw_residual: Whether to use yaw residual ("auto", True, False)
            max_vacuum_pressure: Maximum vacuum pressure for normalization
            deterministic: Whether to use deterministic policy
            clip_final_by_step: Whether to clip final actions by step size
        """
        self.num_chambers = num_chambers
        self.rl_available = rl_available
        self.rl_controller = None
        self.heuristic_controller = None
        self.use_rl = False
        
        # Initialize heuristic controller (always available)
        if rl_available:
            try:
                self.heuristic_controller = HeuristicController(
                    mode=heuristic_mode,
                    num_chambers=num_chambers,
                    rotation_angle=0.0,
                    step_size=heuristic_step_size,
                    delta_yaw=heuristic_delta_yaw,
                    damping=heuristic_damping,
                )
            except Exception as e:
                rospy.logwarn(f"Could not initialize heuristic controller: {e}")
                self.heuristic_controller = None
        
        # Initialize RL controller if model path is provided
        if model_path and rl_available:
            try:
                self.rl_controller = ResidualRLController(
                    model_path=model_path,
                    num_chambers=num_chambers,
                    heuristic_mode=heuristic_mode,
                    heuristic_step_size=heuristic_step_size,
                    heuristic_delta_yaw=heuristic_delta_yaw,
                    heuristic_damping=heuristic_damping,
                    step_lateral=step_lateral,
                    step_yaw=step_yaw,
                    residual_scale_xy=residual_scale_xy,
                    residual_scale_yaw=residual_scale_yaw,
                    use_yaw_residual=use_yaw_residual,
                    max_vacuum_pressure=max_vacuum_pressure,
                    deterministic=deterministic,
                    clip_final_by_step=clip_final_by_step,
                )
                self.use_rl = True
                rospy.loginfo(f"RL controller initialized with model: {model_path}")
            except Exception as e:
                rospy.logerr(f"Failed to initialize RL controller: {e}")
                self.rl_controller = None
                self.use_rl = False
        else:
            if not rl_available:
                rospy.logwarn("RL controllers not available, using heuristic only")
            else:
                rospy.loginfo("No model path provided, using heuristic only")
    
    def get_model_path(self, chamber_count: int, model_type: str = "hgreedy") -> str:
        """
        Get the model path for a specific chamber count and model type.
        
        Args:
            chamber_count: Number of chambers (3, 4, 5, or 6)
            model_type: Type of model ("hgreedy", "hmomentum", "hyaw", "hyaw_momentum")
            
        Returns:
            Full path to the model file
        """
        model_dir = "/home/edg/catkin_ws/src/suction_cup/models"
        model_filename = f"ch{chamber_count}_{model_type}_last_model_mlp.zip"
        return os.path.join(model_dir, model_filename)
    
    def load_model(self, chamber_count: int, model_type: str = "hgreedy") -> bool:
        """
        Load a specific RL model.
        
        Args:
            chamber_count: Number of chambers (3, 4, 5, or 6)
            model_type: Type of model ("hgreedy", "hmomentum", "hyaw", "hyaw_momentum")
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not rl_available:
            rospy.logerr("RL controllers not available")
            return False
        
        model_path = self.get_model_path(chamber_count, model_type)
        
        if not os.path.exists(model_path):
            rospy.logerr(f"Model file not found: {model_path}")
            return False
        
        try:
            # Reinitialize with new model
            self.rl_controller = ResidualRLController(
                model_path=model_path,
                num_chambers=chamber_count,
                heuristic_mode="greedy" if "greedy" in model_type else "momentum",
                heuristic_step_size=0.5,
                heuristic_delta_yaw=1.0,
                heuristic_damping=0.9,
                step_lateral=1.0,
                step_yaw=3.0,
                residual_scale_xy=0.5,
                residual_scale_yaw=0.5,
                use_yaw_residual="auto",
                max_vacuum_pressure=3.1171e4,
                deterministic=True,
                clip_final_by_step=True,
            )
            self.use_rl = True
            self.num_chambers = chamber_count
            rospy.loginfo(f"Successfully loaded RL model: {model_path}")
            return True
        except Exception as e:
            rospy.logerr(f"Failed to load RL model: {e}")
            self.rl_controller = None
            self.use_rl = False
            return False
    
    def compute_action(self, 
                      vacuum_pressures: np.ndarray, 
                      rotation_angle_deg: float,
                      return_debug: bool = False) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Compute action using RL controller or heuristic fallback.
        
        Args:
            vacuum_pressures: Array of vacuum pressures from chambers
            rotation_angle_deg: Current rotation angle in degrees
            return_debug: Whether to return debug information
            
        Returns:
            Tuple of (lateral_velocity, yaw_velocity, debug_info)
        """
        if self.use_rl and self.rl_controller:
            try:
                # Use RL controller
                lateral_vel, yaw_vel = self.rl_controller.compute_action(
                    vacuum_pressures=vacuum_pressures,
                    rotation_angle_deg=rotation_angle_deg,
                    return_debug=return_debug
                )
                
                debug_info = {
                    "controller_type": "RL",
                    "model_loaded": True,
                    "lateral_velocity": lateral_vel,
                    "yaw_velocity": yaw_vel,
                }
                
                if return_debug and len(self.rl_controller.compute_action(
                    vacuum_pressures, rotation_angle_deg, return_debug=True)) == 3:
                    _, _, rl_debug = self.rl_controller.compute_action(
                        vacuum_pressures, rotation_angle_deg, return_debug=True)
                    debug_info.update(rl_debug)
                
                return lateral_vel, yaw_vel, debug_info
                
            except Exception as e:
                rospy.logwarn(f"RL controller failed, falling back to heuristic: {e}")
                self.use_rl = False
        
        # Fallback to heuristic controller
        if self.heuristic_controller:
            try:
                self.heuristic_controller.rotation_angle = float(rotation_angle_deg)
                lateral_vel, yaw_vel = self.heuristic_controller.compute_action(vacuum_pressures)
                
                debug_info = {
                    "controller_type": "Heuristic",
                    "model_loaded": False,
                    "lateral_velocity": lateral_vel,
                    "yaw_velocity": yaw_vel,
                }
                
                return lateral_vel, yaw_vel, debug_info
                
            except Exception as e:
                rospy.logerr(f"Heuristic controller failed: {e}")
                # Return zero action as last resort
                return np.array([0.0, 0.0]), 0.0, {"controller_type": "None", "error": str(e)}
        else:
            rospy.logerr("No controller available")
            return np.array([0.0, 0.0]), 0.0, {"controller_type": "None", "error": "No controller available"}
    
    def reset_history(self):
        """Reset the history buffer for RL controller."""
        if self.use_rl and self.rl_controller:
            try:
                self.rl_controller.reset_history()
            except Exception as e:
                rospy.logwarn(f"Failed to reset RL controller history: {e}")
    
    def get_available_models(self) -> Dict[int, list]:
        """
        Get list of available models for each chamber count.
        
        Returns:
            Dictionary mapping chamber count to list of available model types
        """
        model_dir = "/home/edg/catkin_ws/src/suction_cup/models"
        available_models = {}
        
        for chamber_count in [3, 4, 5, 6]:
            models = []
            for model_type in ["hgreedy", "hmomentum", "hyaw", "hyaw_momentum"]:
                model_path = self.get_model_path(chamber_count, model_type)
                if os.path.exists(model_path):
                    models.append(model_type)
            available_models[chamber_count] = models
        
        return available_models
    
    def is_rl_available(self) -> bool:
        """Check if RL controllers are available."""
        return self.rl_available
    
    def is_model_loaded(self) -> bool:
        """Check if an RL model is currently loaded."""
        return self.use_rl and self.rl_controller is not None


# Convenience function for easy integration
def create_rl_controller(chamber_count: int, 
                        model_type: str = "hgreedy",
                        **kwargs) -> RLControllerHelper:
    """
    Create an RL controller helper with a specific model.
    
    Args:
        chamber_count: Number of chambers (3, 4, 5, or 6)
        model_type: Type of model ("hgreedy", "hmomentum", "hyaw", "hyaw_momentum")
        **kwargs: Additional arguments for RLControllerHelper
        
    Returns:
        Initialized RLControllerHelper instance
    """
    helper = RLControllerHelper(num_chambers=chamber_count, **kwargs)
    
    if helper.is_rl_available():
        success = helper.load_model(chamber_count, model_type)
        if not success:
            rospy.logwarn(f"Failed to load model ch{chamber_count}_{model_type}, using heuristic only")
    else:
        rospy.logwarn("RL not available, using heuristic only")
    
    return helper


if __name__ == "__main__":
    # Test the RL controller helper
    rospy.init_node('rl_controller_test')
    
    # Test with different chamber counts
    for ch in [3, 4, 5, 6]:
        print(f"\nTesting chamber count: {ch}")
        helper = create_rl_controller(ch, "hgreedy")
        
        # Test with dummy pressure data
        dummy_pressures = np.array([-1000, -2000, -1500, -1800][:ch])
        dummy_angle = 45.0
        
        lateral_vel, yaw_vel, debug = helper.compute_action(dummy_pressures, dummy_angle, return_debug=True)
        print(f"Lateral velocity: {lateral_vel}")
        print(f"Yaw velocity: {yaw_vel}")
        print(f"Debug info: {debug}")
