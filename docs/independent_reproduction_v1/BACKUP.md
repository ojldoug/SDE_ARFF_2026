# Transfer the v1 bundle to an independent destination

No off-KW61146 backup is currently verified. The 6,978,959,360-byte bundle (6.50 GiB) is identified by `BUNDLE.json`. Another directory on KW61146 does not qualify. No transfer has been executed by this verification task.

From the destination machine (replace `SOURCE_HOST` and paths with your authorized SSH endpoint and storage):

```sh
mkdir -p /independent/storage/sde-v1
rsync --partial --progress SOURCE_HOST:/home/kammonaa/projects/SDE_ARFF/artifact_exports/independent_reproduction_v1.tar /independent/storage/sde-v1/
# Use the versioned verification script from this repository on the destination:
bash scripts/verify_artifact_backup.sh /independent/storage/sde-v1/independent_reproduction_v1.tar > /independent/storage/sde-v1/VERIFIED.txt
```

The script checks SHA-256 `54c939eb8d8add8ca92e3aa1c5906cc7379803124dd642ac426d69b9a06a51a4` and exits nonzero on mismatch. It records destination hostname and verification time. Retain `BUNDLE.json`, `artifact_manifest.json`, this repository commit, and `VERIFIED.txt` alongside the destination copy. Confirm the destination is separate hardware/storage, then record that location and receipt privately; credentials must not enter Git. A checksum verifies copy integrity, not storage durability or access permissions. No public URL or upload authorization is implied.

For additional payload validation on the destination, with a checkout of the release:

```sh
python scripts/package_independent_reproduction.py verify-tar --archive /independent/storage/sde-v1/independent_reproduction_v1.tar
```

Only after destination checksum success may the copy be called a verified backup. Public hosting remains a separate decision.

Closure check (22 September 2026): the source bundle checksum was rechecked. No destination verification receipt or public hosting identifier is recorded in the release records. `SOURCE_HOST` must be the author's authorized SSH endpoint for KW61146; destination storage must be on another system. This procedure was prepared, not executed.
