#!/usr/bin/env python

# Authors: Jungpyo Lee
# Create: Sep. 30. 2025
# Description: RL controller helper for integrating 2D Haptic Search models into ROS-based haptic search system.
#              This module provides a bridge between the ResidualRLController and the ROS haptic search framework.

import os
import sys
import numpy as np
from typing import Tuple, Dict, Any, Optional
from collections import deque
import rospy

# Add the 2D Haptic Search project to the path
haptic_search_path = "/home/edg/Desktop/Jungpyo/Github/2D_Haptic_Search"
if haptic_search_path not in sys.path:
    sys.path.append(haptic_search_path)

try:
    from stable_baselines3 import PPO
    rl_available = True
except ImportError as e:
    rospy.logwarn(f"Could not import PPO from stable_baselines3: {e}")
    rl_available = False

def calculate_unit_vectors(self, num_chambers, yaw_angle):
    """
    Calculate the unit vectors for the suction cup.
    Note that depending on the number of chambers, the unit vectors could be different. Please check the endeffector.
    """
    return [np.array([np.cos(2 * np.pi / (num_chambers * 2) + 2 * np.pi * i / num_chambers),
                    np.sin(2 * np.pi / (num_chambers * 2) + 2 * np.pi * i / num_chambers)])
        for i in range(num_chambers)]

def calculate_direction_vector(self, unit_vectors, vacuum_pressures):
    direction_vector = np.sum([vp * uv for vp, uv in zip(vacuum_pressures, unit_vectors)], axis=0)
    return direction_vector / np.linalg.norm(direction_vector) if np.linalg.norm(direction_vector) > 0 else np.array([0, 0])

class HeuristicController:
    def __init__(self, mode, num_chambers, rotation_angle, step_size=0.5, delta_yaw=1.5, damping=0.7):
        self.mode = mode
        self.step_size = step_size
        self.num_chambers = num_chambers
        self.rotation_angle = rotation_angle
        self.delta_yaw = delta_yaw
        self.damping = damping
        self.velocity = np.array([0.0, 0.0])  # for momentum mode

    def compute_action(self, vacuum_pressures):
        unit_vectors = calculate_unit_vectors(self.num_chambers, self.rotation_angle)
        direction = calculate_direction_vector(unit_vectors, vacuum_pressures)

        if self.mode == "greedy":
            velocity = self.step_size * direction
            delta_yaw = 0

        elif self.mode == "momentum":
            self.velocity = self.damping * self.velocity + self.step_size * direction
            velocity = self.velocity
            delta_yaw = 0

        elif self.mode == "yaw":
            velocity = self.step_size * direction
            delta_yaw = self.delta_yaw
        
        elif self.mode == "yaw_momentum":
            self.velocity = self.damping * self.velocity + self.step_size * direction
            velocity = self.velocity
            delta_yaw = self.delta_yaw

        else:
            raise ValueError(f"Unknown controller mode: {self.mode}")

        return velocity, delta_yaw



class ResidualRLController:
    """
    Runtime controller for residual RL (per-ch model). Composes heuristic + residual.
    Observation fed to the policy must match training:
      base_obs = [ pressures_norm (0..1)^N , heuristic_preview_norm ([-1,1]^(2 or 3)) ]
    If the model was trained with history stacking of length K, the true policy input is:
      obs = concat( base_obs_{t-K+1}, ..., base_obs_t )  in R^{K * base_obs_dim}
    This controller auto-detects K from the loaded model and stacks internally.

    Action: policy outputs residual in [-1,1]^k; we scale, clip, and add to heuristic.
    """

    def __init__(
        self,
        model_path: str,
        num_chambers: int,
        heuristic_mode: str = "greedy",
        heuristic_step_size: float = 0.5,
        heuristic_delta_yaw: float = 1.5,
        heuristic_damping: float = 0.7,
        step_lateral: float = 1.0,
        step_yaw: float = 3.0,
        residual_scale_xy: float = 1.0,
        residual_scale_yaw: float = 1.0,
        use_yaw_residual: bool | str = "auto",   # allow "auto"
        max_vacuum_pressure: float = 3.1171e4,
        deterministic: bool = True,
        clip_final_by_step: bool = True,
    ):
        # Load model with CPU device and custom objects for compatibility
        import torch
        # Remove .zip extension if present to avoid .zip.zip duplication
        clean_path = model_path
        if clean_path.endswith('.zip'):
            clean_path = clean_path[:-4]
        self.model = PPO.load(clean_path, device='cpu')
        self.deterministic = bool(deterministic)

        # Env/physics config
        self.num_chambers = int(num_chambers)
        self.max_vacuum_pressure = float(max_vacuum_pressure)

        self.step_lateral = float(step_lateral)
        self.step_yaw = float(step_yaw)

        self.residual_scale_xy = float(residual_scale_xy)
        self.residual_scale_yaw = float(residual_scale_yaw)
        self.clip_final_by_step = bool(clip_final_by_step)

        # ---------- Detect yaw-residual ON/OFF from shapes ----------
        exp_obs_dim = int(np.prod(self.model.observation_space.shape))
        exp_act_dim = int(np.prod(self.model.action_space.shape))
        cand_noyaw_obs = self.num_chambers + 2
        cand_yaw_obs   = self.num_chambers + 3

        if use_yaw_residual == "auto":
            # try to deduce from action dim primarily
            if exp_act_dim == 3:
                assumed_yaw = True
            elif exp_act_dim == 2:
                assumed_yaw = False
            else:
                raise ValueError(f"[ResidualRLController] Unexpected action dim: {exp_act_dim}")
            self.use_yaw_residual = assumed_yaw
        else:
            self.use_yaw_residual = bool(use_yaw_residual)

        base_obs_dim = (self.num_chambers + (3 if self.use_yaw_residual else 2))
        base_act_dim = (3 if self.use_yaw_residual else 2)

        # ---------- Infer history length K and wrapper type ----------
        # Try HistoryActionWrapper FIRST (obs + action stacking) - more specific
        if exp_obs_dim % (base_obs_dim + base_act_dim) == 0:
            K = exp_obs_dim // (base_obs_dim + base_act_dim)
            self.history_K = int(K)
            self.use_action_history = True   # HistoryActionWrapper
        # Try HistoryWrapper second (obs only stacking) - more general
        elif exp_obs_dim % base_obs_dim == 0:
            K = exp_obs_dim // base_obs_dim
            self.history_K = int(K)
            self.use_action_history = False  # HistoryWrapper
        else:
            # If mismatch, try flipping yaw assumption as a last resort
            alt_base_obs = (self.num_chambers + (2 if self.use_yaw_residual else 3))
            alt_base_act = (2 if self.use_yaw_residual else 3)
            
            # Try HistoryWrapper with flipped yaw
            if exp_obs_dim % alt_base_obs == 0:
                self.use_yaw_residual = not self.use_yaw_residual
                base_obs_dim = alt_base_obs
                base_act_dim = alt_base_act
                K = exp_obs_dim // base_obs_dim
                self.history_K = int(K)
                self.use_action_history = False
            # Try HistoryActionWrapper with flipped yaw
            elif exp_obs_dim % (alt_base_obs + alt_base_act) == 0:
                self.use_yaw_residual = not self.use_yaw_residual
                base_obs_dim = alt_base_obs
                base_act_dim = alt_base_act
                K = exp_obs_dim // (base_obs_dim + base_act_dim)
                self.history_K = int(K)
                self.use_action_history = True
            else:
                # Check if this might be a chamber count mismatch
                expected_chambers = None
                if exp_obs_dim == 45:  # chamber 3 with HistoryActionWrapper
                    expected_chambers = 3
                elif exp_obs_dim == 50:  # chamber 4 with HistoryActionWrapper
                    expected_chambers = 4
                elif exp_obs_dim == 55:  # chamber 5 with HistoryActionWrapper
                    expected_chambers = 5
                elif exp_obs_dim == 60:  # chamber 6 with HistoryActionWrapper
                    expected_chambers = 6
                elif exp_obs_dim == 30:  # chamber 3 with HistoryWrapper
                    expected_chambers = 3
                elif exp_obs_dim == 35:  # chamber 4 with HistoryWrapper
                    expected_chambers = 4
                elif exp_obs_dim == 40:  # chamber 5 with HistoryWrapper
                    expected_chambers = 5
                elif exp_obs_dim == 45:  # chamber 6 with HistoryWrapper
                    expected_chambers = 6
                
                if expected_chambers:
                    raise ValueError(
                        f"[ResidualRLController] Chamber count mismatch: model was trained with {expected_chambers} chambers "
                        f"(obs_dim={exp_obs_dim}) but you're trying to use it with {self.num_chambers} chambers. "
                        f"Please use the correct model for {self.num_chambers} chambers. "
                        f"Model path: {model_path}"
                    )
                else:
                    raise ValueError(
                        f"[ResidualRLController] Model/args mismatch: model obs_dim={exp_obs_dim}, act_dim={exp_act_dim}, "
                        f"but with num_chambers={self.num_chambers} valid combinations are:\n"
                        f"  - HistoryWrapper: K * {cand_noyaw_obs} (no-yaw) or K * {cand_yaw_obs} (yaw)\n"
                        f"  - HistoryActionWrapper: K * ({cand_noyaw_obs}+{2}) (no-yaw) or K * ({cand_yaw_obs}+{3}) (yaw)\n"
                        f"Cannot factor obs_dim into any valid pattern."
                    )

        # Validate action dim against yaw mode
        want_act = 3 if self.use_yaw_residual else 2
        if exp_act_dim != want_act:
            raise ValueError(
                f"[ResidualRLController] Action dim mismatch: model act_dim={exp_act_dim}, "
                f"but use_yaw_residual={self.use_yaw_residual} -> expected {want_act}."
            )

        # Heuristic baseline
        self.heuristic = HeuristicController(
            mode=heuristic_mode,
            num_chambers=self.num_chambers,
            rotation_angle=0.0,
            step_size=float(heuristic_step_size),
            delta_yaw=float(heuristic_delta_yaw),
            damping=float(heuristic_damping),
        )

        # Internal history buffer (stores base_obs, NOT normalized obs)
        self._obs_hist = deque(maxlen=self.history_K)
        # For HistoryActionWrapper: also store action history
        self._act_hist = deque(maxlen=self.history_K) if self.use_action_history else None
        self._last_action = None  # Store last action for HistoryActionWrapper
        self._initialized = False  # to bootstrap history on first call

    # ---------- public API ----------
    def compute_action(
        self,
        vacuum_pressures: np.ndarray,
        rotation_angle_deg: float,
        return_debug: bool = False,
    ) -> Tuple[np.ndarray, float] | Tuple[np.ndarray, float, Dict[str, Any]]:
        # 1) normalize pressures
        p = np.array(vacuum_pressures, dtype=np.float32)
        p_norm = np.clip(p / self.max_vacuum_pressure, 0.0, 1.0)

        # 2) heuristic baseline (env units)
        self.heuristic.rotation_angle = float(rotation_angle_deg)
        vel_h, dyaw_h = self.heuristic.compute_action(vacuum_pressures)
        if self.use_yaw_residual:
            a_h = np.array([vel_h[0], vel_h[1], dyaw_h], dtype=float)
        else:
            a_h = np.array([vel_h[0], vel_h[1]], dtype=float)

        # 3) heuristic preview (normalized)
        hprev = self._normalize_hpreview(a_h)

        # 4) build base_obs (current time)
        base_obs = np.concatenate([p_norm, hprev], axis=0).astype(np.float32)

        # 5) stack history -> policy obs (auto K)
        obs = self._stack_obs(base_obs, self._last_action)

        # 6) predict residual
        a_r_raw, _ = self.model.predict(obs, deterministic=self.deterministic)
        a_r_raw = np.array(a_r_raw, dtype=float)

        # 7) compose final action
        a_final, a_r_applied = self._compose_final(a_h, a_r_raw)

        # 8) optional final clipping
        if self.clip_final_by_step:
            a_final[0] = float(np.clip(a_final[0], -self.step_lateral, self.step_lateral))
            a_final[1] = float(np.clip(a_final[1], -self.step_lateral, self.step_lateral))
            if a_final.shape[0] == 3:
                a_final[2] = float(np.clip(a_final[2], -self.step_yaw, self.step_yaw))

        dx, dy = float(a_final[0]), float(a_final[1])
        dtheta = float(a_final[2]) if a_final.shape[0] == 3 else 0.0

        # Update last action for HistoryActionWrapper
        if self.use_action_history:
            self._last_action = a_final.copy()

        if return_debug:
            debug = {
                "history_K": self.history_K,
                "base_obs_dim": base_obs.shape[0],
                "policy_obs": obs,
                "obs_pressures_norm": p_norm,
                "obs_hpreview_norm": hprev,
                "a_h": a_h,
                "a_r_raw": a_r_raw,
                "a_r_applied": a_r_applied,
                "a_final": a_final,
                "use_action_history": self.use_action_history,
            }
            return np.array([dx, dy], dtype=float), dtheta, debug
        return np.array([dx, dy], dtype=float), dtheta

    # ---------- helpers ----------
    def _normalize_hpreview(self, a_h: np.ndarray) -> np.ndarray:
        hx = np.clip(a_h[0] / self.step_lateral, -1.0, 1.0)
        hy = np.clip(a_h[1] / self.step_lateral, -1.0, 1.0)
        if a_h.shape[0] == 3:
            hth = np.clip(a_h[2] / self.step_yaw, -1.0, 1.0)
            return np.array([hx, hy, hth], dtype=np.float32) if self.use_yaw_residual else np.array([hx, hy], dtype=np.float32)
        return np.array([hx, hy], dtype=np.float32)

    def _compose_final(self, a_h: np.ndarray, a_r_raw: np.ndarray):
        if self.use_yaw_residual:
            if a_r_raw.shape[0] != 3:
                raise ValueError("Residual head expects 3 dims.")
            dx_r = np.clip(a_r_raw[0] * self.residual_scale_xy * self.step_lateral, -self.step_lateral, self.step_lateral)
            dy_r = np.clip(a_r_raw[1] * self.residual_scale_xy * self.step_lateral, -self.step_lateral, self.step_lateral)
            dth_r = np.clip(a_r_raw[2] * self.residual_scale_yaw * self.step_yaw, -self.step_yaw, self.step_yaw)
            a_r = np.array([dx_r, dy_r, dth_r], dtype=float)
            return a_h + a_r, a_r
        else:
            if a_r_raw.shape[0] != 2:
                raise ValueError("Residual head expects 2 dims.")
            dx_r = np.clip(a_r_raw[0] * self.residual_scale_xy * self.step_lateral, -self.step_lateral, self.step_lateral)
            dy_r = np.clip(a_r_raw[1] * self.residual_scale_xy * self.step_lateral, -self.step_lateral, self.step_lateral)
            a_r = np.array([dx_r, dy_r], dtype=float)
            if a_h.shape[0] == 3:  # yaw pass-through
                return np.array([a_h[0] + dx_r, a_h[1] + dy_r, a_h[2]], dtype=float), np.array([dx_r, dy_r, 0.0], dtype=float)
            return a_h + a_r, a_r

    # ---------- history stacking ----------
    def _stack_obs(self, base_obs: np.ndarray, prev_action: np.ndarray = None) -> np.ndarray:
        """
        Maintain a deque of base_obs (length K) and return the concatenated vector.
        For K=1 (history-free models), this returns base_obs unchanged.
        For HistoryActionWrapper, also maintain action history and interleave obs+action.
        """
        if not self._initialized:
            # Bootstrap history on first call
            self._obs_hist.clear()
            if self._act_hist is not None:
                self._act_hist.clear()
            
            if self.history_K > 1:
                # Fill K-1 frames with zeros
                zero_obs = np.zeros_like(base_obs, dtype=np.float32)
                zero_act = np.zeros(3 if self.use_yaw_residual else 2, dtype=np.float32)
                for _ in range(self.history_K - 1):
                    self._obs_hist.append(zero_obs)
                    if self._act_hist is not None:
                        self._act_hist.append(zero_act)
            
            self._obs_hist.append(base_obs.astype(np.float32))
            if self._act_hist is not None:
                # Use zero action for first frame
                zero_act = np.zeros(3 if self.use_yaw_residual else 2, dtype=np.float32)
                self._act_hist.append(zero_act)
            self._initialized = True
        else:
            self._obs_hist.append(base_obs.astype(np.float32))
            if self._act_hist is not None and prev_action is not None:
                self._act_hist.append(prev_action.astype(np.float32))

        if self.history_K == 1:
            return base_obs
        
        if self.use_action_history:
            # HistoryActionWrapper: interleave obs and action
            stacked = []
            for i in range(self.history_K):
                if i < len(self._obs_hist):
                    stacked.append(self._obs_hist[i])
                if i < len(self._act_hist):
                    stacked.append(self._act_hist[i])
            return np.concatenate(stacked, axis=0).astype(np.float32)
        else:
            # HistoryWrapper: only observations
            return np.concatenate(list(self._obs_hist), axis=0).astype(np.float32)

    # Optional: allow external reset of the internal history (e.g., at episode start)
    def reset_history(self, fill: str = "zeros"):
        """Call this at episode boundaries if your integration needs explicit reset."""
        self._initialized = False
        self._last_action = None




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
                heuristic_damping=0.7,
                step_lateral=1.0,
                step_yaw=3.0,
                residual_scale_xy=1.0,
                residual_scale_yaw=1.0,
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

