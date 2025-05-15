#!/bin/bash
#SBATCH --job-name=project1
#SBATCH --account=a-large-sc
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:00:30           # ≥ 10 s
#SBATCH --signal=USR1@10            # send only to batch
#SBATCH --output=a
#SBATCH --error=b
#SBATCH --export=ALL,MAX_ITER=3000000

set -euo pipefail

echo "[wrapper] Job $SLURM_JOB_ID  host=$SLURM_NODELIST"

export CKPT_DIR=/iopsstor/scratch/cscs/$USER/checkpoints
mkdir -p "$CKPT_DIR"

# checkpoint (< MAX_ITER)
latest=""
for f in "$CKPT_DIR"/ckpt_*.txt; do
  [[ -e $f ]] || break
  iter=${f##*ckpt_}; iter=${iter%%.txt}; iter=$((10#$iter))
  if (( iter < MAX_ITER )); then
    [[ -z $latest || $f -nt $latest ]] && latest=$f
  fi
done
[[ -n $latest ]] && RESUME="--resume $latest" || RESUME=""

# signal forwarding
child_pid=0
forward_usr1() {
  echo "[wrapper] SIGUSR1 caught → forwarding to srun task" >&2
  [[ $child_pid -ne 0 ]] && kill -USR1 "$child_pid"
}
trap forward_usr1 SIGUSR1

# launch
set +e
srun bash -c 'python3 compute2.py --max-iter "$MAX_ITER" '"$RESUME" \
     2> >(grep -Ev 'Exited with exit code 1|Terminating StepId' >&2) &
child_pid=$!
wait "$child_pid"
status=$?
set -e

echo "[wrapper] compute2.py exit status=$status"

# chain or stop
if [[ $status -eq 0 ]]; then
  echo "[wrapper] Timeout checkpoint → resubmitting…" >&2
  sbatch "$0"
elif [[ $status -eq 1 ]]; then
  echo "[wrapper] Reached MAX_ITER ($MAX_ITER) → chain ends." >&2
else
  echo "[wrapper] compute failed (status=$status) → aborting chain." >&2
fi

exit 0
