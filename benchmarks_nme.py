import numpy as np
import math as m
import argparse
import logging
import os
import pandas as pd

"""
path/                                    # main folder
    --0550/                              # person number
        --graund_truth.npy               # ground truth points file
        --data/                          # folder with compared data
            --detector1.npy       
            --detector2.npy
            --detector3.npy
            --detector4.npy
"""


def print_args(arguments):
    for arg in vars(arguments):
        s = arg + ': ' + str(getattr(arguments, arg))
        logging.info(s)


def calculate_nme(gt_lands, checked_lands, dist, n_points):
    points_errors = []
    for gt, ch in zip(gt_lands, checked_lands):
        x0, y0 = gt
        x1, y1 = ch
        point_error = m.sqrt((x1 - x0)**2 + (y1 - y0)**2) / dist
        points_errors.append(point_error)
    return np.sum(points_errors) / n_points


def compare_detectors(gt_path, ch_path, idx, bench):
    gt = np.load(gt_path)

    if len(bench) != 0:
        bench["id"].append(idx)
    else:
        bench["id"] = [idx]

    for filename in os.listdir(ch_path):
        if filename.endswith(".npy"):
            name = filename.replace("image_train_", "").replace(".npy", "")
            name = name.replace(idx+"_", "")
            checked = np.load(os.path.join(ch_path, filename))
            nme = calculate_nme(gt, checked, args.distance, args.points_number)
            # logging.info(f"NME for {name} at {idx}: {nme}")
            if name not in detector_list:
                detector_list.append(name)
                bench[name] = [nme]
            else:
                bench[name].append(nme)
        else:
            continue


def get_benchmark():
    gt_path = None
    ch_path = None
    benchmark = {}
    for i, person in enumerate(os.listdir(args.path)):
        if person.startswith("."):
            continue
        else:
            for path in os.listdir(os.path.join(args.path, person)):
                if os.path.isdir(os.path.join(args.path, person, path)):
                    ch_path = os.path.join(args.path, person, path)
                elif path.endswith(".npy"):
                    gt_path = os.path.join(args.path, person, path)
                else:
                    continue
            compare_detectors(gt_path, ch_path, person, benchmark)
    total_benchmark = {}
    for k, key in enumerate(benchmark):
        if k == 0:
            continue
        total_benchmark[key] = np.mean(benchmark[key])
    logging.info(f"Total NME: {total_benchmark}")
    bdf = pd.DataFrame.from_dict(benchmark)
    bdf.to_csv(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default='',
                        type=str)
    parser.add_argument('-d', '--distance',
                        default=450,
                        help='equal size')
    parser.add_argument('--points_number',
                        default=68,
                        help='number of points')
    parser.add_argument('--det_number',
                        default=4,
                        help='number of compared detectors')
    parser.add_argument('-o', '--output',
                        default='result.csv',
                        help='path to save the table of results')
    args = parser.parse_args()
    # print_args(args)

    logging.basicConfig(level=logging.INFO)

    detector_list = []

    get_benchmark()
