#!/bin/bash

# The running and pending Slurm jobs of all users, and in brackets those of
# one user: monitor.sh [<user>], by default the current user.
RUNNING=$(squeue | grep "1 gpu" | wc -l)
PENDING=$(squeue | grep "PD" | wc -l)

JOBUSER="${1:-$USER}"

RUNNINGUSR=$(squeue -u "$JOBUSER" | grep "1 gpu" | wc -l)
PENDINGUSR=$(squeue -u "$JOBUSER" | grep "PD" | wc -l)

echo "Running Processes: $RUNNING ($RUNNINGUSR)"
echo "Pending Processes: $PENDING ($PENDINGUSR)"
