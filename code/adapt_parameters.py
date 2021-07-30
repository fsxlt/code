# -*- coding: utf-8 -*-
from os.path import join
import argparse


def get_args():
    ROOT_DIRECTORY = "./"
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, "data/")
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, "checkpoint_adapt")

    parser = argparse.ArgumentParser()
    parser.add_argument("--override", type=str2bool, default=True)

    parser.add_argument("--experiment", type=str, default="debug")
    parser.add_argument("--ptl", type=str, default="bert")
    parser.add_argument("--model", type=str, default="bert-base-uncased")

    parser.add_argument("--dataset_name", type=str, default="mldoc")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--adapt_trn_languages", type=str, default="english")

    # fast adaptation setup
    parser.add_argument("--adapt_epochs", type=int, default=5)
    parser.add_argument("--adapt_batch_size", type=int, default=32)
    parser.add_argument("--adapt_num_shots", type=int, default=5)
    parser.add_argument("--adapt_optimizer", type=str, default="adam")
    parser.add_argument("--adapt_lr", type=float, default=5e-5)
    parser.add_argument("--group_index", type=int, default=-1)

    # training setup
    parser.add_argument("--train_all_params", type=str2bool, default=True)
    parser.add_argument("--train_classifier", type=str2bool, default=True)
    parser.add_argument("--train_pooler", type=str2bool, default=True)
    parser.add_argument("--reinit_classifier", type=str2bool, default=False)

    # speeding up inference
    parser.add_argument("--inference_batch_size", type=int, default=512)

    # init with ckpt for adaptation, if not, "scratch"
    parser.add_argument("--load_ckpt", type=str2bool, default=False)
    parser.add_argument("--ckpt_path", type=str, default="")

    # early stoping
    parser.add_argument("--early_stop", type=str2bool, default=False)
    parser.add_argument("--early_stop_patience", type=int, default=2)

    # miscs
    parser.add_argument("--data_path", default=RAW_DATA_DIRECTORY, type=str)
    parser.add_argument("--checkpoint", default=TRAINING_DIRECTORY, type=str)
    parser.add_argument("--manual_seed", type=int, default=3, help="manual seed")
    parser.add_argument("--summary_freq", default=100, type=int)
    parser.add_argument("--time_stamp", default=None, type=str)
    parser.add_argument("--train_fast", default=True, type=str2bool)
    parser.add_argument("--track_time", default=True, type=str2bool)
    parser.add_argument("--world", default="0", type=str)

    # parse conf.
    conf = parser.parse_args()
    return conf


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    args = get_args()
