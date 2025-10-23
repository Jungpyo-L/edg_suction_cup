#!/usr/bin/env python3

"""
Example usage of the new joint limit checking functions in rtde_helper.py

This example shows how to use the new joint limit validation functions
that were added to rtde_helper.py for use in experiments.
"""

import rospy
from geometry_msgs.msg import PoseStamped
from helperFunction.rtde_helper import rtdeHelp

def example_joint_limit_usage():
    """
    Example showing how to use the new joint limit checking functions.
    """
    # Initialize ROS (required for rtde_helper)
    rospy.init_node('joint_limit_example')
    
    # Create RTDE helper
    rtde_help = rtdeHelp(125)
    
    print("=== Joint Limit Checking Functions Example ===\n")
    
    # Example 1: Check joint limits directly
    print("1. Checking joint limits directly:")
    joint_positions = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]  # Example joint angles
    joints_valid = rtde_help.checkJointLimits(joint_positions)
    print(f"   Joint positions {joint_positions}")
    print(f"   Are within limits: {joints_valid}\n")
    
    # Example 2: Check pose safety
    print("2. Checking pose safety:")
    pose_vector = [0.5, 0.0, 0.3, 0.0, 0.0, 0.0]  # [x, y, z, rx, ry, rz]
    pose_safe = rtde_help.checkPoseSafety(pose_vector)
    print(f"   Pose vector {pose_vector}")
    print(f"   Is safe: {pose_safe}\n")
    
    # Example 3: Validate a complete pose (recommended method)
    print("3. Validating a complete pose (recommended):")
    
    # Create a PoseStamped object
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = 0.5
    pose.pose.position.y = 0.0
    pose.pose.position.z = 0.3
    pose.pose.orientation.x = 0.0
    pose.pose.orientation.y = 0.0
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 1.0
    
    # Validate the pose
    pose_valid = rtde_help.validatePose(pose)
    print(f"   Pose: x={pose.pose.position.x}, y={pose.pose.position.y}, z={pose.pose.position.z}")
    print(f"   Is valid and reachable: {pose_valid}\n")
    
    # Example 4: Using in pose generation loop
    print("4. Using in pose generation loop:")
    print("   This is how it's used in generate_random_initial_pose():")
    print("   ```python")
    print("   for attempt in range(max_attempts):")
    print("       # Generate random pose...")
    print("       random_pose = generate_random_pose()")
    print("       ")
    print("       # Validate pose")
    print("       if rtde_help.validatePose(random_pose):")
    print("           return random_pose  # Valid pose found")
    print("       else:")
    print("           print(f'Pose {attempt + 1} failed validation, trying again...')")
    print("   ```\n")
    
    print("=== Benefits of the new functions ===")
    print("✓ Reusable across different experiments")
    print("✓ Clean separation of concerns")
    print("✓ Easy to use and understand")
    print("✓ Handles all RTDE validation internally")
    print("✓ Returns simple boolean results")

if __name__ == "__main__":
    try:
        example_joint_limit_usage()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
