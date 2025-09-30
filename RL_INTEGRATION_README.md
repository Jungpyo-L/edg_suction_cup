# RL Controller Integration for 2D Haptic Search

This document describes the integration of RL models from the 2D Haptic Search project into the ROS-based haptic search system.

## Overview

The integration provides a bridge between the `ResidualRLController` from the 2D Haptic Search project and the existing ROS-based haptic search framework. This allows you to use trained RL models for haptic search control while maintaining compatibility with the existing system.

## Files Added/Modified

### New Files
- `src/helperFunction/RL_controller_helper.py` - Main integration module
- `src/test_rl_integration.py` - Test script for the integration
- `src/example_rl_usage.py` - Usage examples
- `RL_INTEGRATION_README.md` - This documentation

### Modified Files
- `src/JP_2D_haptic_search_hopping.py` - Updated to support RL controllers

## Available RL Models

The following RL models are available in `models/` directory:

### Chamber 3 Models
- `ch3_hgreedy_last_model_mlp.zip` - Greedy heuristic + RL residual
- `ch3_hmomentum_last_model_mlp.zip` - Momentum heuristic + RL residual
- `ch3_hyaw_last_model_mlp.zip` - Greedy + yaw + RL residual
- `ch3_hyaw_momentum_last_model_mlp.zip` - Momentum + yaw + RL residual

### Chamber 4 Models
- `ch4_hgreedy_last_model_mlp.zip`
- `ch4_hmomentum_last_model_mlp.zip`
- `ch4_hyaw_last_model_mlp.zip`
- `ch4_hyaw_momentum_last_model_mlp.zip`

### Chamber 5 Models
- `ch5_hgreedy_last_model_mlp.zip`
- `ch5_hmomentum_last_model_mlp.zip`
- `ch5_hyaw_last_model_mlp.zip`
- `ch5_hyaw_momentum_last_model_mlp.zip`

### Chamber 6 Models
- `ch6_hgreedy_last_model_mlp.zip`
- `ch6_hmomentum_last_model_mlp.zip`
- `ch6_hyaw_last_model_mlp.zip`
- `ch6_hyaw_momentum_last_model_mlp.zip`

## Usage

### 1. Using RL Controllers in Haptic Search

To use an RL controller, specify it in the `--controller` argument:

```bash
# Use RL controller with greedy heuristic
python JP_2D_haptic_search_hopping.py --controller rl_hgreedy --ch 4

# Use RL controller with momentum heuristic
python JP_2D_haptic_search_hopping.py --controller rl_hmomentum --ch 4

# Use RL controller with yaw control
python JP_2D_haptic_search_hopping.py --controller rl_hyaw --ch 4

# Use RL controller with momentum and yaw
python JP_2D_haptic_search_hopping.py --controller rl_hyaw_momentum --ch 4
```

### 2. Programmatic Usage

```python
from helperFunction.RL_controller_helper import create_rl_controller

# Create RL controller
rl_controller = create_rl_controller(
    chamber_count=4,
    model_type="hgreedy"
)

# Use in haptic search loop
pressures = np.array([-1000, -2000, -1500, -1800])
yaw_angle = 45.0

lateral_vel, yaw_vel, debug = rl_controller.compute_action(
    vacuum_pressures=pressures,
    rotation_angle_deg=yaw_angle,
    return_debug=True
)

# Reset for new episode
rl_controller.reset_history()
```

### 3. Available Controller Types

- `normal` - Original heuristic controller (greedy)
- `yaw` - Heuristic controller with yaw rotation
- `momentum` - Heuristic controller with momentum
- `momentum_yaw` - Heuristic controller with momentum and yaw
- `rl_hgreedy` - RL controller with greedy heuristic baseline
- `rl_hmomentum` - RL controller with momentum heuristic baseline
- `rl_hyaw` - RL controller with greedy + yaw baseline
- `rl_hyaw_momentum` - RL controller with momentum + yaw baseline

## Testing

### Run Integration Tests
```bash
cd /home/edg/catkin_ws/src/suction_cup/src
python test_rl_integration.py
```

### Run Usage Examples
```bash
cd /home/edg/catkin_ws/src/suction_cup/src
python example_rl_usage.py
```

## Dependencies

The integration requires the following dependencies:

1. **2D Haptic Search Project**: Located at `/home/edg/Desktop/Jungpyo/Github/2D_Haptic_Search`
2. **Stable Baselines3**: For loading PPO models
3. **PyTorch**: For neural network inference
4. **NumPy**: For numerical operations
5. **ROS**: For the main haptic search system

## Architecture

### RLControllerHelper Class

The `RLControllerHelper` class provides:

- **Model Loading**: Automatic loading of RL models based on chamber count and type
- **Fallback Support**: Falls back to heuristic controller if RL fails
- **History Management**: Handles history stacking for RL models
- **Debug Information**: Provides detailed debug information for analysis

### Integration Points

1. **Model Path Resolution**: Automatically finds models in `models/` directory
2. **Action Conversion**: Converts RL outputs to transformation matrices
3. **Error Handling**: Graceful fallback to heuristic controllers
4. **History Reset**: Proper history management for episode boundaries

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the 2D Haptic Search project is in the correct path
2. **Model Not Found**: Check that model files exist in `models/` directory
3. **RL Not Available**: Install required dependencies (stable-baselines3, torch)
4. **Chamber Mismatch**: Use the correct model for your chamber count

### Debug Information

The RL controller provides detailed debug information:

```python
lateral_vel, yaw_vel, debug = rl_controller.compute_action(
    pressures, yaw_angle, return_debug=True
)

print(debug)
# Output includes:
# - controller_type: "RL" or "Heuristic"
# - model_loaded: True/False
# - lateral_velocity: [x, y] velocity
# - yaw_velocity: rotation velocity
# - Additional RL-specific debug info
```

## Performance Considerations

1. **Model Loading**: Models are loaded once at initialization
2. **Inference Speed**: RL inference is typically fast (< 1ms)
3. **Memory Usage**: Models require additional memory for neural network weights
4. **History Buffer**: History stacking uses additional memory for state storage

## Future Enhancements

1. **Model Selection**: Automatic model selection based on performance
2. **Online Learning**: Real-time model updates during operation
3. **Multi-Model Ensemble**: Using multiple models for improved performance
4. **Adaptive Parameters**: Dynamic parameter adjustment based on performance
