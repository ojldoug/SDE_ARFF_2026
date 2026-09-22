"""No numerical code: portable archive roundtrip, deduplication and corruption gate."""
import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/package_independent_reproduction.py'
class PackageTests(unittest.TestCase):
 def test_roundtrip_and_bad_manifest(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);data=b'science-input\n';files=[]
   for name in ['data/a.bin','data/b.bin']:
    p=root/name;p.parent.mkdir(exist_ok=True);p.write_bytes(data);files.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
   manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(files=files)));archive=root/'payload.tar'
   base=[sys.executable,str(SCRIPT)]
   subprocess.run(base+['bundle','--root',str(root),'--manifest',str(manifest),'--output',str(archive)],check=True,capture_output=True)
   subprocess.run(base+['verify-tar','--manifest',str(manifest),'--archive',str(archive)],check=True,capture_output=True)
   files[0]['sha256']='0'*64;manifest.write_text(json.dumps(dict(files=files)))
   result=subprocess.run(base+['verify-tar','--manifest',str(manifest),'--archive',str(archive)],capture_output=True)
   self.assertNotEqual(result.returncode,0)
   # Existing tar must never be replaced, even with valid inputs.
   files[0]['sha256']=hashlib.sha256(data).hexdigest();manifest.write_text(json.dumps(dict(files=files)))
   self.assertNotEqual(subprocess.run(base+['bundle','--root',str(root),'--manifest',str(manifest),'--output',str(archive)],capture_output=True).returncode,0)
if __name__=='__main__':unittest.main()
