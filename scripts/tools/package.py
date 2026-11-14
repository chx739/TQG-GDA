import argparse
from pathlib import Path
from experiments.common.archives import package

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args();r=package(args.source,args.archive)
    print(f"Archived and verified {r['files_verified']} files; SHA256 {r['archive_sha256']}")


if __name__ == "__main__":
    main()
