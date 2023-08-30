import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='community_area', type=str)  # original
parser.add_argument("--year_start", default='14', type=str)
parser.add_argument("--year_end", default='19', type=str)
parser.add_argument("--version", default='v1', type=str)
parser.add_argument("--train_bool", default=1, type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--batchsize", default=128, type=int)
parser.add_argument("--learning_rate", default=2e-3, type=float) #2e-4  #2e-5
parser.add_argument("--dropout", default=0.8, type=float)
parser.add_argument("--embed_dim", default=50, type=int)
parser.add_argument("--prob_duplicate", default=0.0, type=float)
parser.add_argument("--start", default=100, type=int)
parser.add_argument("--end", default=115, type=int)
parser.add_argument("--device", default='cpu:0', type=str)

args = parser.parse_args()


def load_parameters():
    FIXED_PARAMETERS = {
        "data_path": "./data/",
        "year_start": args.year_start,
        "year_end": args.year_end,
        "dataset": args.dataset,
        "train_bool": args.train_bool,
        "NAME": 'grid_best',# 'weights_'+args.year_start+"_"+args.year_end+"_"+args.dataset+"_"+args.version,
        "version": args.version,
        "epochs": args.epochs,
        "batchsize": args.batchsize,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "embed_dim": args.embed_dim,
        "gamma":2.0,
        "alpha":0.9,
        "prob_duplicate": args.prob_duplicate,
        "start": args.start,
        "end": args.end,
        "device": args.device
    }
    return FIXED_PARAMETERS
