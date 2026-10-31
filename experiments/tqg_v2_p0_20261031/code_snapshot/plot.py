from p0_summarize import summarize
import argparse

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",required=True)
    summarize(parser.parse_args().stage)
