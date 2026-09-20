"""New capacity-regime study: reuse frozen numerical runners and worker machinery."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_controlled_ex8_float64_v2 as adapter
import run_controlled_campaign_float64_v2 as campaign
STUDY=ROOT/'results/capacity_regime_ex8_v2'
adapter.STUDY=adapter.base.STUDY=campaign.STUDY=STUDY
