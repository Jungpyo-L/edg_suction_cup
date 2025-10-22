#!/usr/bin/env python

# Example usage of RL controller integration
# This shows how to use the RL controller in your haptic search system

import sys
import os
import numpy as np

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_usage():
    """Example of how to use the RL controller in haptic search."""
    
    print("RL Controller Integration Example")
    print("=" * 40)
    
    try:
        from helperFunction.RL_controller_helper import create_rl_controller
        
        # Example 1: Create RL controller for 4-chamber suction cup
        print("\n1. Creating RL controller for 4-chamber suction cup...")
        rl_controller = create_rl_controller(
            chamber_count=4,
            model_type="hgreedy"  # or "hmomentum", "hyaw", "hyaw_momentum"
        )
        
        print(f"   RL available: {rl_controller.is_rl_available()}")
        print(f"   Model loaded: {rl_controller.is_model_loaded()}")
        
        # Example 2: Simulate haptic search loop
        print("\n2. Simulating haptic search loop...")
        
        # Simulate pressure readings from 4 chambers
        pressure_readings = [
            np.array([-1000, -2000, -1500, -1800]),  # Initial reading
            np.array([-1200, -1800, -1600, -1900]),  # After movement
            np.array([-1400, -1600, -1700, -2000]),  # Getting closer
            np.array([-1600, -1400, -1800, -2100]),  # Almost there
            np.array([-1800, -1200, -1900, -2200]),  # Success!
        ]
        
        yaw_angles = [0.0, 15.0, 30.0, 45.0, 60.0]
        
        for i, (pressures, yaw) in enumerate(zip(pressure_readings, yaw_angles)):
            print(f"\n   Step {i+1}:")
            print(f"   Pressures: {pressures}")
            print(f"   Yaw angle: {yaw}°")
            
            # Compute action using RL controller
            lateral_vel, yaw_vel, debug = rl_controller.compute_action(
                vacuum_pressures=pressures,
                rotation_angle_deg=yaw,
                return_debug=True
            )
            
            print(f"   Lateral velocity: {lateral_vel}")
            print(f"   Yaw velocity: {yaw_vel:.3f}")
            print(f"   Controller type: {debug.get('controller_type', 'unknown')}")
            
            # Check if we've reached success condition (all pressures below threshold)
            success_threshold = -2000
            if all(p < success_threshold for p in pressures):
                print("   SUCCESS: All pressures below threshold!")
                break
        
        # Example 3: Reset for new episode
        print("\n3. Resetting controller for new episode...")
        rl_controller.reset_history()
        print("   Controller history reset")
        
        print("\n" + "=" * 40)
        print("Example completed successfully!")
        
    except Exception as e:
        print(f"Error in example: {e}")
        import traceback
        traceback.print_exc()

def show_available_models():
    """Show available RL models."""
    print("\nAvailable RL Models:")
    print("=" * 20)
    
    try:
        from helperFunction.RL_controller_helper import RLControllerHelper
        
        helper = RLControllerHelper()
        available_models = helper.get_available_models()
        
        for chamber_count, models in available_models.items():
            if models:
                print(f"Chamber {chamber_count}: {', '.join(models)}")
            else:
                print(f"Chamber {chamber_count}: No models available")
                
    except Exception as e:
        print(f"Error getting available models: {e}")

if __name__ == "__main__":
    # Show available models
    show_available_models()
    
    # Run example
    example_usage()

