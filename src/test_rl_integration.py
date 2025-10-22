#!/usr/bin/env python

# Test script for RL controller integration
# This script tests the RL controller helper without requiring ROS

import sys
import os
import numpy as np

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_rl_controller_helper():
    """Test the RL controller helper functionality."""
    print("Testing RL Controller Helper...")
    
    try:
        from helperFunction.RL_controller_helper import RLControllerHelper, create_rl_controller
        
        # Test 1: Check if RL is available
        helper = RLControllerHelper(num_chambers=4)
        print(f"RL available: {helper.is_rl_available()}")
        
        # Test 2: Check available models
        available_models = helper.get_available_models()
        print(f"Available models: {available_models}")
        
        # Test 3: Try to load a model
        if available_models.get(4, []):
            model_type = available_models[4][0]  # Use first available model
            print(f"Trying to load model: ch4_{model_type}")
            
            success = helper.load_model(4, model_type)
            print(f"Model loaded successfully: {success}")
            
            if success:
                # Test 4: Compute action
                dummy_pressures = np.array([-1000, -2000, -1500, -1800])
                dummy_angle = 45.0
                
                lateral_vel, yaw_vel, debug = helper.compute_action(
                    dummy_pressures, dummy_angle, return_debug=True
                )
                
                print(f"Lateral velocity: {lateral_vel}")
                print(f"Yaw velocity: {yaw_vel}")
                print(f"Debug info: {debug}")
                
                # Test 5: Reset history
                helper.reset_history()
                print("History reset successfully")
        
        print("RL Controller Helper test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing RL Controller Helper: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_controller_creation():
    """Test the convenience function for creating controllers."""
    print("\nTesting controller creation...")
    
    try:
        from helperFunction.RL_controller_helper import create_rl_controller
        
        # Test creating controllers for different chamber counts
        for ch in [3, 4, 5, 6]:
            print(f"\nTesting chamber count: {ch}")
            helper = create_rl_controller(ch, "hgreedy")
            
            # Test with dummy data
            dummy_pressures = np.array([-1000, -2000, -1500, -1800][:ch])
            dummy_angle = 30.0
            
            lateral_vel, yaw_vel, debug = helper.compute_action(
                dummy_pressures, dummy_angle, return_debug=True
            )
            
            print(f"  Lateral velocity: {lateral_vel}")
            print(f"  Yaw velocity: {yaw_vel}")
            print(f"  Controller type: {debug.get('controller_type', 'unknown')}")
        
        print("Controller creation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing controller creation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("RL Controller Integration Test")
    print("=" * 50)
    
    # Test 1: Basic functionality
    test1_success = test_rl_controller_helper()
    
    # Test 2: Controller creation
    test2_success = test_controller_creation()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  RL Controller Helper: {'PASS' if test1_success else 'FAIL'}")
    print(f"  Controller Creation: {'PASS' if test2_success else 'FAIL'}")
    print("=" * 50)
    
    if test1_success and test2_success:
        print("All tests passed! RL integration is working correctly.")
        sys.exit(0)
    else:
        print("Some tests failed. Check the error messages above.")
        sys.exit(1)

