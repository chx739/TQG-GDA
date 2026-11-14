"""Create and verify a durable, nonoverwriting local experiment archive."""
import argparse
import hashlib
from pathlib import Path
import tarfile
from experiments.common.synthetic.archive import digest, manifest, write_json


def package(source, destination):
    source=source.resolve();destination=destination.resolve()
    if source in destination.parents:
        raise ValueError('archive must live outside the source directory')
    files=manifest(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive open prevents accidental destruction of previous runs.
    with destination.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as archive:
            archive.add(source,arcname=source.name,recursive=True)
    checked=0
    with tarfile.open(destination,'r:gz') as archive:
        for member in archive:
            if member.isdir():continue
            if not member.isfile():raise ValueError(f'Unexpected nonregular archive member: {member.name}')
            relative=str(Path(member.name).relative_to(source.name))
            expected=files[relative]
            h=hashlib.sha256()
            stream=archive.extractfile(member)
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            if member.size!=expected['bytes'] or h.hexdigest()!=expected['sha256']:
                raise ValueError(f'Archive round-trip mismatch: {relative}')
            checked+=1
    assert checked==len(files)
    record=dict(archive_path=str(destination),archive_bytes=destination.stat().st_size,
                archive_sha256=digest(destination),source_path=str(source),
                files_verified=checked,uncompressed_bytes=sum(v['bytes'] for v in files.values()),
                member_manifest=files)
    write_json(destination.with_suffix(destination.suffix+'.json'),record)
    with destination.with_suffix(destination.suffix+'.sha256').open('x') as f:
        f.write(record['archive_sha256']+'  '+destination.name+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args();r=package(args.source,args.archive)
    print(f"Archived and verified {r['files_verified']} files; SHA256 {r['archive_sha256']}")


if __name__=='__main__':main()
