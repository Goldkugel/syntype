RUNNING=$(squeue | grep "1 gpu" | wc -l)
PENDING=$(squeue | grep "PD" | wc -l)

USER="pallaoro"

RUNNINGUSR=$(squeue -u $USER | grep "1 gpu" | wc -l)
PENDINGUSR=$(squeue -u $USER | grep "PD" | wc -l)

echo "Running Processes: $RUNNING ($RUNNINGUSR)"
echo "Pending Processes: $PENDING ($PENDINGUSR)"
