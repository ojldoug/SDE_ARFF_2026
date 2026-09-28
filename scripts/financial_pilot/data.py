"""Causal hourly features; sealed test is opened only with a recorded gate."""
from pathlib import Path
import datetime as dt,json,hashlib,csv,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'results/financial_covariance_pilot_v1'
ASSETS=['BTCUSDT','ETHUSDT','BNBUSDT']
def timestamp(s):return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(test=False):
 if test and not (P/'PRETEST_GATE.json').exists():raise RuntimeError('Sealed test prices inaccessible before gate')
 start=timestamp('2021-12-01')+3600;stop=timestamp('2026-01-01' if test else '2025-01-01')
 t=np.arange(start,stop+1,3600,dtype=np.int64);close=np.full((len(t),3),np.nan);manifest=json.loads((P/'DATA_MANIFEST.json').read_text())
 for row in manifest['records']:
  if row['month'].startswith('2025') and not test:continue
  path=ROOT/row['path'];assert sha(path)==row['sha256'];unit=1000000 if row['timestamp_unit']=='microsecond' else 1000
  k=ASSETS.index(row['asset'])
  with zipfile.ZipFile(path) as z:
   for v in csv.reader(z.read(z.namelist()[0]).decode().splitlines()):
    op=int(v[0]);cl=int(v[6])
    if cl!=op+3600*unit-1:continue # recorded incomplete bar, same acquisition exclusion
    end=(cl+1)//unit;i=(end-start)//3600;assert t[i]==end
    value=float(v[4]);assert np.isfinite(value) and value>0
    if np.isfinite(close[i,k]):raise ValueError('Duplicate aligned close')
    close[i,k]=value
 r=np.full_like(close,np.nan);r[1:]=np.log(close[1:])-np.log(close[:-1]);return t,r

def make_features(t,r):
 indices=[];features=[]
 for j in range(168,len(r)):
  w=r[j-168:j]
  if not np.isfinite(w).all() or not np.isfinite(r[j]).all():continue
  means=w.mean(0);center=w-means;var=np.mean(center*center,0);corr=[]
  for a,b in [(0,1),(0,2),(1,2)]:corr.append(0. if min(var[a],var[b])<=1e-16 else float(np.clip(np.mean(center[:,a]*center[:,b])/np.sqrt(var[a]*var[b]),-1,1)))
  features.append(np.r_[w[-1],np.log(np.maximum(np.mean(w[-24:]**2,0),1e-16)),np.log(np.maximum(np.mean(w*w,0),1e-16)),corr]);indices.append(j)
 idx=np.asarray(indices);X=np.asarray(features);year=np.array([dt.datetime.fromtimestamp(int(t[i]-1),dt.timezone.utc).year for i in idx]);return idx,X,year

def prepare():
 out=P/'development.npz'
 if out.exists():raise FileExistsError(out)
 t,r=load();idx,X,year=make_features(t,r);fit=(year>=2022)&(year<=2023);val=year==2024
 expected={'fit':17520,'validation':8784};counts={'fit':int(fit.sum()),'validation':int(val.sum())}
 # Test eligibility gate uses timestamps only, never prices.
 structural=json.loads((P/'STRUCTURAL_TIMESTAMPS.json').read_text());sets={a:set() for a in ASSETS}
 for row in structural:sets[row['asset']].update(row['interval_ends'])
 common=set.intersection(*sets.values());test_eligible=sum(all(end-3600*j in common for j in range(170)) for end in range(timestamp('2025-01-01')+3600,timestamp('2026-01-01')+1,3600))
 counts['test_structural']=int(test_eligible);expected['test_structural']=8760
 for k in counts:
  if counts[k]/expected[k]<.95:raise ValueError('Eligibility below95%: '+k)
 mu=X[fit].mean(0);sd=X[fit].std(0);sd[sd<1e-12]=1.;D=np.sqrt(np.mean(r[idx[fit]]**2,axis=0));assert np.all(D>0)
 np.savez_compressed(out,t=t,r=r,idx=idx,X=X,year=year,fit=fit,validation=val,mu=mu,sd=sd,D=D)
 with (P/'forecast_timestamps_development.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['row','split','forecast_origin_UTC_seconds','latest_predictor_timestamp','target_interval_start','target_interval_end'])
  for a,j in enumerate(idx):
   assert t[j]-t[j-1]==3600
   w.writerow([int(j),'fit' if fit[a] else 'validation' if val[a] else 'warmup',int(t[j-1]),int(t[j-1]),int(t[j-1]),int(t[j])])
 record={'counts':counts,'expected':expected,'eligible_fraction':{k:counts[k]/expected[k] for k in counts},'excluded':{k:expected[k]-counts[k] for k in counts},'development_sha256':sha(out),'DATA_MANIFEST_sha256':sha(P/'DATA_MANIFEST.json'),'preprocessing':dict(feature_mean=mu.tolist(),feature_sd=sd.tolist(),return_scale_D=D.tolist()),'test_price_values_accessed':False}
 (P/'SPLIT_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':prepare()
