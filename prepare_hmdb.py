import argparse

from stpr.splits import make_hmdb_lists


def main():
    parser = argparse.ArgumentParser(description="Prepare HMDB51 official split-2 lists")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", default="data/hmdb51")
    args = parser.parse_args()
    make_hmdb_lists(args.splits, args.output)


if __name__ == "__main__":
    main()
