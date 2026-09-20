#!/usr/bin/env python3
"""GPU worker entrypoint: do not import CPU-only supervisor modules here."""
import run_controlled_ex8_float64_v2 as adapter
adapter.STUDY=adapter.base.STUDY=adapter.ROOT/'results/capacity_regime_ex8_final_h_v1'
if __name__=='__main__':adapter.main()
