import numpy as np
import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
import pdb, sys, traceback
# from ipdb import launch_ipdb_on_exception

def main(args):
    if args.tst_free:
        trainer = FSLTrainer(args)
        trainer.train()
    else:
        trainer = FSLTrainer(args)
        trainer.train()
        trainer.evaluate_test(use_max_tst=True)

    print(args.save_path)

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    # with launch_ipdb_on_exception():
    pprint(vars(args))

    # We will set CUDA_VISIBLE_DEVICES in the terminal
    #set_gpu(args.gpu)

    if args.debug_fast:
        try:
            main(args)
        except:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    else:
        main(args)


