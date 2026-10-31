from d34_summarize import summarize
import argparse
if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",required=True)
    summarize(p.parse_args().stage)
