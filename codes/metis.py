import numpy as np
import pymetis
import argparse

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser("Multi-Level GRL")

    parser.add_argument("--output_path", type=str, default="./libra")
    parser.add_argument("--partition", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="flickr")

    n_cuts, membership = pymetis.part_graph(partition, adjacency=adjacency_list)