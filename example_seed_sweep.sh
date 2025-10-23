#!/bin/bash

# Example script showing how to run experiments with multiple seeds
# This will run the same experiment 3 times with seeds 42, 43, and 44

echo "Running polygon grid experiment with seed sweep: 42, 43, 44"
echo "This will run the same experiment 3 times in sequence"
echo ""

# Run experiment with multiple seeds
python3 JP_2D_polygon_grid_experiment.py \
    --ch 4 \
    --controllers greedy rl_hyaw_momentum \
    --seeds 42 43 44 \
    --max_iterations 50 \
    --pause_time 0.2 \
    --hop_sleep 0.15

echo ""
echo "All experiments completed!"
echo "Check ~/SuctionExperiment/YYMMDD/ for results"
