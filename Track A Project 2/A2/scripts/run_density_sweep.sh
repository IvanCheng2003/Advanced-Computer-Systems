#!/usr/bin/env bash
set -e

M=2048
K=2048
N=2048
THREADS=4
REPS=5

OUT_CSV="density_sweep.csv"

echo "m,k,n,density,kernel,threads,best_time_s,GFLOP_s,nnz" > "$OUT_CSV"

for D in 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5; do
    echo "Running density=$D ..."
    ./bench_gemm_spmm $M $K $N $D dense $THREADS $REPS >> "$OUT_CSV"
    ./bench_gemm_spmm $M $K $N $D spmm  $THREADS $REPS >> "$OUT_CSV"
done
