#!/usr/bin/env python3
"""One-shot detached verification wrapper; never retries or changes numerical code."""
import argparse, fcntl, json, os, runpy, subprocess, sys, time, traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['supervise','child'])
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--resources',type=Path,required=True)
    p.add_argument('--forbidden-tree',type=Path,required=True)
    p.add_argument('--lock-dir',type=Path,required=True)
    p.add_argument('--gpu',default='0')
    a=p.parse_args(); w=a.workspace.resolve(); resources=a.resources.resolve(); root=resources/'checkout'
    if a.mode=='child':
        forbidden=a.forbidden_tree.resolve(); reads=set()
        def audit(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
                path=Path(os.fsdecode(args[0])).absolute()
                if path.is_relative_to(forbidden):
                    raise RuntimeError('Forbidden original-tree access: '+str(path))
                if path.is_relative_to(root): reads.add(str(path.relative_to(root)))
        sys.addaudithook(audit)
        sys.path.insert(0,str(root/'scripts'))
        sys.argv=[str(root/'scripts/reproduction_route.py'),'baseline','--method','arff','--seed','0','--execute','--output',str(w/'run')]
        try:runpy.run_path(sys.argv[0],run_name='__main__')
        finally:
            modules={k:str(v.__file__) for k,v in sys.modules.copy().items() if getattr(v,'__file__',None)}
            (w/'isolation.json').write_text(json.dumps(dict(original_tree_reads_blocked=True,checkout_files_opened=sorted(reads),module_paths=modules,sys_path=sys.path),indent=2))
        return
    marker=w/'STARTED.json'
    with marker.open('x') as f:json.dump(dict(started=time.time(),supervisor_pid=os.getpid()),f)
    state={}
    def save(name,obj):(w/name).write_text(json.dumps(obj,indent=2)+'\n')
    try:
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='4c5dc6b54af216ebba49477874c955d259d4140f'
        subprocess.run(['git','diff','--exit-code'],cwd=root,check=True)
        fd=os.open(a.lock_dir/f'gpu_{a.gpu}.lock',os.O_CREAT|os.O_RDWR,0o600)
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def smi(*args):return subprocess.check_output(['nvidia-smi','-i',a.gpu,*args],text=True).strip()
        assert not smi('--query-compute-apps=pid','--format=csv,noheader'),'GPU occupied'
        assert int(smi('--query-gpu=memory.used','--format=csv,noheader,nounits'))<512
        env=dict(os.environ)
        for k in ['PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','XLA_FLAGS','JAX_COMPILATION_CACHE_DIR']:env.pop(k,None)
        env.update(CUDA_VISIBLE_DEVICES=a.gpu,JAX_PLATFORMS='cuda',JAX_ENABLE_X64='false',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        cmd=[str(resources/'venv/bin/python'),'-B',str(Path(__file__).resolve()),'child','--workspace',str(w),'--resources',str(resources),'--forbidden-tree',str(a.forbidden_tree),'--lock-dir',str(a.lock_dir),'--gpu',a.gpu]
        state=dict(status='running',command=cmd,gpu=a.gpu,gpu_details=smi('--query-gpu=name,uuid,driver_version','--format=csv,noheader'),environment={k:env[k] for k in env if k.startswith(('JAX','XLA','CUDA','OMP','OPENBLAS','PYTHON'))},peak_gpu_memory_MiB=0,start=time.time())
        with (w/'console.log').open('x') as log,(w/'gpu_memory.csv').open('x') as mem:
            mem.write('elapsed_seconds,gpu_memory_MiB\n')
            proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,))
            state['child_pid']=proc.pid
            while proc.poll() is None:
                used=int(smi('--query-gpu=memory.used','--format=csv,noheader,nounits'))
                state['peak_gpu_memory_MiB']=max(state['peak_gpu_memory_MiB'],used)
                mem.write(f'{time.time()-state["start"]},{used}\n');mem.flush()
                save('status.json',state);time.sleep(1)
        state.update(returncode=proc.returncode,process_wall_seconds=time.time()-state['start'])
        if proc.returncode:raise RuntimeError('Single authorized fit failed; no automatic retry')
        state['status']='execution_complete_comparison_pending';save('EXECUTION_COMPLETE.json',state);save('status.json',state)
    except Exception:
        state.update(status='failed',failure=traceback.format_exc());save('FAILURE.json',state);save('status.json',state);raise
if __name__=='__main__':main()
