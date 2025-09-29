#!/usr/bin/env bash
set -euo pipefail

# Pin execution to core 0 for consistency
PIN="taskset -c 0"

# -------------------
# Batch A: Zero-queue latency
# -------------------
$PIN ./Proj2 --mode latency --min 4K --max 1G --points 20 --stride 64B --csv A_latency_64B.csv
sleep 5

# -------------------
# Batch B: Pattern × stride sweep
# -------------------
for STR in 64B 256B 1024B; do
  $PIN ./Proj2 --mode bw --min 64K --max 1G --points 10 --stride $STR --pattern seq --csv B_seq_${STR}.csv
  $PIN ./Proj2 --mode bw --min 64K --max 1G --points 10 --stride $STR --pattern random --csv B_random_${STR}.csv
  sleep 3
done

# -------------------
# Batch C: Read/Write mix sweep
# -------------------
for MIX in 100R 100W 70R30W 50R50W; do
  $PIN ./Proj2 --mode mix --size 512M --stride 64B --rw $MIX --csv C_mix_${MIX}.csv
  sleep 3
done

# -------------------
# Batch D: Intensity sweep
# -------------------
for T in 1 2 4 8; do
  $PIN ./Proj2 --mode intensity --size 1G --stride 64B --threads $T --csv D_intensity_T${T}.csv
  sleep 3
done

# -------------------
# Batch E: Working-set size sweep (cache locality transitions)
# -------------------
$PIN ./Proj2 --mode locality --min 4K --max 1G --points 20 --stride 64B --csv E_locality.csv
sleep 5

# -------------------
# Batch F: Cache-miss impact (light kernel, e.g., multiply)
# -------------------
for STR in 64B 4K; do
  $PIN ./Proj2 --mode cache --min 64K --max 1G --points 12 --pattern seq --stride $STR --csv F_cache_seq_${STR}.csv
  $PIN ./Proj2 --mode cache --min 64K --max 1G --points 12 --pattern random --stride $STR --csv F_cache_random_${STR}.csv
  sleep 3
done

# -------------------
# Batch G: TLB impact
# -------------------
$PIN ./Proj2 --mode tlb --min 4M --max 2G --points 8 --pattern random --csv G_tlb_baseline.csv
sudo $PIN ./Proj2 --mode tlb --min 4M --max 2G --points 8 --pattern random --huge true --csv G_tlb_huge.csv
