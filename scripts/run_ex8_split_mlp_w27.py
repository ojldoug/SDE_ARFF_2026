#!/usr/bin/env python3
"""Fixed modern joint-versus-split ablation; accepted numerical modules unchanged."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_ex8_mlp_w27 import effective_config

def check_assigned_gpu(allow_cpu=False):
    if allow_cpu:
        if os.environ.get('JAX_PLATFORMS')!='cpu':raise RuntimeError('CPU smoke requires explicit CPU backend')
        return
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ['0','1','2','3']:raise RuntimeError('Exactly one GPU index must be visible')
    result=subprocess.check_output(['nvidia-smi','-i',visible,'--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    for line in result.splitlines():
        if line.strip() and int(line.split(',')[0].strip())!=os.getpid():raise RuntimeError('Assigned GPU occupied: '+line)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--artifact-path',type=Path);p.add_argument('--inspect',action='store_true');args=p.parse_args(argv)
    if not 0<=args.seed<2**32:raise ValueError('Seed range')
    config=effective_config('ex8')
    if args.inspect:
        print(json.dumps(dict(effective_config=asdict(config),hidden_sizes=[27,27],drift_parameters=893,covariance_parameters=921,total_parameters=1814,formulation='MSE drift; fivefold OOF residual Gaussian NLL factor covariance;300epochs per regression',timing='four-GPU non-isolated accuracy'),indent=2));return
    path=args.artifact_path
    if path is None or path.suffix!='.npz' or path.exists():raise ValueError('New .npz artifact path required')
    import run_adam_split_mlp_experiment as runner
    original=(runner.get_config,runner.check_machine_idle,sys.argv)
    runner.get_config=effective_config
    # Scheduling-only change: each of the four exclusive workers checks its
    # assigned GPU; unrelated jobs on that GPU are never bypassed.
    runner.check_machine_idle=check_assigned_gpu
    sys.argv=[runner.__file__,'ex8','--seed',str(args.seed),'--artifact-path',str(path.resolve())]
    try:runner.main()
    finally:runner.get_config,runner.check_machine_idle,sys.argv=original
if __name__=='__main__':main()
