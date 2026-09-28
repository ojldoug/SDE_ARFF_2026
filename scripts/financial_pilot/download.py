"""Public spot monthly archives only. Does not inspect price values."""
from pathlib import Path
import urllib.request,hashlib,json,datetime,concurrent.futures,time,zipfile,csv
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'results/financial_covariance_pilot_v1'
ASSETS=['BTCUSDT','ETHUSDT','BNBUSDT'];MONTHS=['2021-12']+[f'{y}-{m:02}' for y in range(2022,2026) for m in range(1,13)]
def fetch(asset,month):
 name=f'{asset}-1h-{month}.zip';url=f'https://data.binance.vision/data/spot/monthly/klines/{asset}/1h/{name}';folder=P/('sealed_test_archives' if month.startswith('2025') else 'development_archives');folder.mkdir(exist_ok=True)
 p=folder/name;cp=folder/(name+'.CHECKSUM')
 for attempt in range(3):
  try:
   expected=urllib.request.urlopen(url+'.CHECKSUM',timeout=60).read()
   if p.exists():data=p.read_bytes()
   else:data=urllib.request.urlopen(url,timeout=60).read()
   digest=hashlib.sha256(data).hexdigest();assert digest==expected.decode().split()[0],name
   if not p.exists():p.write_bytes(data)
   if not cp.exists():cp.write_bytes(expected)
   # Structural metadata only, no price access/printing; units fixed by calendar.
   unit=1000000 if month.startswith('2025') else 1000;ends=[];invalid=[];total=0
   with zipfile.ZipFile(p) as z:
    assert len(z.namelist())==1
    for row in csv.reader(z.read(z.namelist()[0]).decode().splitlines()):
     if len(row)!=12:raise ValueError('Kline columns')
     op=int(row[0]);cl=int(row[6]);total+=1;assert op% (3600*unit)==0
     assert datetime.datetime.fromtimestamp(op/unit,datetime.timezone.utc).strftime('%Y-%m')==month, 'Ambiguous timestamp units/calendar'
     if cl!=op+3600*unit-1:
      invalid.append(dict(open_time=op,close_time=cl,reason='not a complete hourly interval'));continue
     ends.append((cl+1)//unit)
   duplicates=len(ends)-len(set(ends));assert not duplicates
   import calendar
   y,m=map(int,month.split('-'));expected_rows=calendar.monthrange(y,m)[1]*24
   return dict(asset=asset,month=month,url=url,checksum_url=url+'.CHECKSUM',sha256=digest,bytes=len(data),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),path=str(p.relative_to(ROOT)),timestamp_unit='microsecond' if unit==1000000 else 'millisecond',rows=len(ends),archive_rows=total,invalid_intervals=invalid,expected_rows=expected_rows,missing_hours=expected_rows-len(ends),duplicate_timestamps=duplicates,interval_ends=ends)
  except Exception:
   if attempt==2:raise
   time.sleep(2*(attempt+1))
def main():
 out=P/'DATA_MANIFEST.json'
 if out.exists():raise FileExistsError('Preserve existing acquisition manifest')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  records=list(pool.map(lambda x:fetch(*x),[(a,m) for a in ASSETS for m in MONTHS]))
 (P/'STRUCTURAL_TIMESTAMPS.json').write_text(json.dumps(records)+'\n')
 for r in records:r.pop('interval_ends')
 out.write_text(json.dumps(dict(records=records,scope='147 official monthly spot archives, no price inspection during acquisition',schema=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore'],revision_policy='Fetched official CHECKSUM stored; future changes must fail hash verification, never silently replace. Official README warns of revisions; local retrieval pins this snapshot.',rights='Code README says MIT; raw data redistribution rights not assumed. Downloader and compact derived reports only are intended for Git.'),indent=2)+'\n');print('Verified',len(records),'archives; structural gaps',sum(r['missing_hours'] for r in records),flush=True)
if __name__=='__main__':main()
