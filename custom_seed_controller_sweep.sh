#!/bin/bash

# Custom script for running different controllers for different seeds
# Seed 42: hyaw_momentum only (greedy already done)
# Seeds 43, 44: greedy and hyaw_momentum

echo "Running custom polygon grid experiment:"
echo "  Seed 42: hyaw_momentum only"
echo "  Seeds 43, 44: greedy and hyaw_momentum"
echo ""

# Run hyaw_momentum for seed 42
echo "=== Running Seed 42: hyaw_momentum only ==="
python3 JP_2D_polygon_grid_experiment.py \
    --ch 4 \
    --controllers hyaw_momentum \
    --seeds 42 \
    --max_iterations 50 \
    --pause_time 0.2 \
    --hop_sleep 0.2

echo ""
echo "Seed 42 completed!"
echo ""

# Run greedy and hyaw_momentum for seed 43
echo "=== Running Seed 43: greedy and hyaw_momentum ==="
python3 JP_2D_polygon_grid_experiment.py \
    --ch 4 \
    --controllers greedy hyaw_momentum \
    --seeds 43 \
    --max_iterations 50 \
    --pause_time 0.2 \
    --hop_sleep 0.2

echo ""
echo "Seed 43 completed!"
echo ""

# Run greedy and hyaw_momentum for seed 44
echo "=== Running Seed 44: greedy and hyaw_momentum ==="
python3 JP_2D_polygon_grid_experiment.py \
    --ch 4 \
    --controllers greedy hyaw_momentum \
    --seeds 44 \
    --max_iterations 50 \
    --pause_time 0.2 \
    --hop_sleep 0.2

echo ""
echo "All experiments completed!"
echo "Check ~/SuctionExperiment/YYMMDD/ for results"
