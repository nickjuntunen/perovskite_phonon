#!/bin/bash

SAMPLE="${1:?usage: run_scf.sh SAMPLE}"
INPUTS_DIR="/home/rotskofflab6/Desktop/research/perovskite_phonon/data/qe_inputs"
QE_BIN_DIR="/home/rotskofflab6/Research/qe-7.3.1/bin"
OUT_DIR="/home/rotskofflab6/Desktop/research/perovskite_phonon/out/${SAMPLE}"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

"${QE_BIN_DIR}/pw.x" -in "${INPUTS_DIR}/${SAMPLE}/${SAMPLE}.scf.in" > scf.log