import argparse
import os
import pickle
import time
from steps import trainer, trainer_alt
from config import MyParser
from logging import getLogger
import torch

logger = getLogger(__name__)
logger.info("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

args = MyParser().parse_args()


os.makedirs(args.exp_dir, exist_ok=True)

if args.resume:
    resume = args.resume
    assert(bool(args.exp_dir))
    with open("%s/args.pkl" % args.exp_dir, "rb") as f:
        old_args = pickle.load(f)
    new_args = vars(args)
    old_args = vars(old_args)
    for key in new_args:
        if key not in old_args or old_args[key] != new_args[key]:
            old_args[key] = new_args[key]
    args = argparse.Namespace(**old_args)
    args.resume = resume
else:
    print("\nexp_dir: %s" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)
args.places = False
logger.info(args)

if args.alternate:
    my_trainer = trainer_alt.Trainer(args)
else:
    my_trainer = trainer.Trainer(args)

if args.phase == "validate":
    my_trainer.validate(hide_progress=True)
elif args.phase == "validate_seg":
    my_trainer.validate_seg(hide_progress=True)
else:
    my_trainer.train()