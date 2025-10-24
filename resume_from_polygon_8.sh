#!/bin/bash

# Resume experiment from polygon 8 (polygon_1_3)
# Polygon mapping:
# 1=polygon_0_0, 2=polygon_0_1, 3=polygon_0_2, 4=polygon_0_3
# 5=polygon_1_0, 6=polygon_1_1, 7=polygon_1_2, 8=polygon_1_3
# 9=polygon_2_0, 10=polygon_2_1, 11=polygon_2_2, 12=polygon_2_3

echo "Resuming experiment from polygon 8 (polygon_1_3)"
echo "This will skip polygons 1-7 and start from polygon 8"
echo ""

# Run hyaw_momentum for seed 42 starting from polygon 8
echo "=== Running Seed 42: hyaw_momentum from polygon 8 ==="
python3 JP_2D_polygon_grid_experiment.py \
    --ch 4 \
    --controllers hyaw_momentum \
    --seeds 42 \
    --start_polygon 8 \
    --max_iterations 50 \
    --pause_time 0.2 \
    --hop_sleep 0.2

echo ""
echo "Experiment resumed from polygon 8 completed!"
echo "Check ~/SuctionExperiment/YYMMDD/ for results"
