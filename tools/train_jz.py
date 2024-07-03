"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)

import torch
import logging
import torch.distributed as dist

from pointcept.engines.train import TRAINERS
from pointcept.utils import comm
from datetime import timedelta

# Jean-Zay
try:
    import idr_torch
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error("Failed to import Jean-Zay idr_torch")
    raise e


DEFAULT_TIMEOUT = timedelta(minutes=60)


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    '''
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )
    '''

    assert (
        torch.cuda.is_available()
    ), "cuda is not available. Please check your installation."

    if idr_torch.size > 1:
        try:
            dist.init_process_group(
                backend="NCCL",
                init_method="env://",
                world_size=idr_torch.size,
                rank=idr_torch.rank,
                timeout=DEFAULT_TIMEOUT,
            )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error("Process group URL: env://")
            raise e

        # Setup the local process group (which contains ranks within the same machine)
        assert comm._LOCAL_PROCESS_GROUP is None
        num_machines = idr_torch.num_nodes
        for i in range(num_machines):
            ranks_on_i = list(
                range(i * idr_torch.ntasks_per_node, (i + 1) * idr_torch.ntasks_per_node)
            )
            pg = dist.new_group(ranks_on_i)
            machine_rank = (idr_torch.rank - idr_torch.local_rank) // idr_torch.ntasks_per_node
            if i == machine_rank:
                comm._LOCAL_PROCESS_GROUP = pg

        assert idr_torch.ntasks_per_node <= torch.cuda.device_count()
        torch.cuda.set_device(idr_torch.local_rank)

        # synchronize is needed here to prevent a possible timeout after calling init_process_group
        # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
        comm.synchronize()
    else:
        main_worker(cfg)

    main_worker(cfg)


if __name__ == "__main__":
    main()
