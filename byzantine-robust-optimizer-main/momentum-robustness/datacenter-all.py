import argparse
import os

import numpy as np
import torch
from torch.nn.modules.loss import MSELoss

from codes.aggregator.base import Mean
from codes.aggregator.clipping import Clipping
from codes.aggregator.coordinatewise_median import CM
from codes.aggregator.krum import Krum
from codes.aggregator.rfa import RFA
from codes.aggregator.trimmed_mean import TM
from codes.attacks.alittle import ALittleIsEnoughAttack
from codes.attacks.bitflipping import BitFlippingWorker
from codes.attacks.xie import IPMAttack
from codes.simulators.server import TorchServer
from codes.simulators.simulator import DistributedEvaluator, ParallelTrainer
from codes.simulators.worker import WorkerWithMomentum
from codes.tasks.datacenter import datacenter, get_datacenter_model
from codes.utils import (
    initialize_logger,
    mean_absolute_error,
    root_mean_squared_error,
)


EXP_ID = os.path.splitext(os.path.basename(__file__))[0]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"
DATA_DIR = ROOT_DIR + "datacenter_processed/"
EXP_DIR = ROOT_DIR + f"outputs/{EXP_ID}/"

parser = argparse.ArgumentParser(description="Datacenter avgcpu regression")
parser.add_argument("--use-cuda", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument(
    "--attack", type=str, default="BF", choices=["NA", "BF", "IPM", "ALIE"]
)
parser.add_argument(
    "--agg", type=str, default="cp", choices=["avg", "cm", "cp", "rfa", "tm", "krum"]
)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--test-batch-size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--max-batches-per-epoch", type=int, default=1000)
parser.add_argument("--split-ratio", type=float, default=0.8)
parser.add_argument("--n-workers", type=int, default=20)

args = parser.parse_args()


N_WORKERS = args.n_workers
N_BYZ = 0 if args.attack == "NA" else 11 if args.attack == "IPM" else 5
MOMENTUM = args.momentum
DATACENTER_FILE_IDS = list(range(N_WORKERS))

if N_WORKERS < 1 or N_WORKERS > 100:
    raise ValueError("--n-workers must be between 1 and 100")

if N_BYZ >= N_WORKERS:
    raise ValueError("The number of Byzantine workers must be smaller than --n-workers")

LOG_DIR = (
    EXP_DIR
    + ("debug/" if args.debug else "")
    + f"f{N_BYZ}_{args.attack}_{args.agg}_m{args.momentum}_seed{args.seed}"
)


def _get_aggregator():
    if args.agg == "avg":
        return Mean()

    if args.agg == "cm":
        return CM()

    if args.agg == "cp":
        return Clipping(tau=10, n_iter=1)

    if args.agg == "rfa":
        return RFA(T=3)

    if args.agg == "tm":
        return TM(b=N_BYZ)

    if args.agg == "krum":
        return Krum(n=N_WORKERS, f=N_BYZ, m=1)

    raise NotImplementedError(args.agg)


def initialize_worker(trainer, worker_rank, model, optimizer, loss_func, device, kwargs):
    train_loader = datacenter(
        data_dir=DATA_DIR,
        file_ids=[worker_rank],
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        split_ratio=args.split_ratio,
        seed=args.seed,
        drop_last=True,
        **kwargs,
    )

    if worker_rank < N_BYZ:
        if args.attack == "BF":
            return BitFlippingWorker(
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )

        if args.attack == "IPM":
            attacker = IPMAttack(
                epsilon=0.1,
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )
            attacker.configure(trainer)
            return attacker

        if args.attack == "ALIE":
            attacker = ALittleIsEnoughAttack(
                n=N_WORKERS,
                m=N_BYZ,
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )
            attacker.configure(trainer)
            return attacker

        raise NotImplementedError(f"No such attack {args.attack}")

    return WorkerWithMomentum(
        momentum=MOMENTUM,
        data_loader=train_loader,
        model=model,
        loss_func=loss_func,
        device=device,
        optimizer=optimizer,
        **kwargs,
    )


def main(args):
    initialize_logger(LOG_DIR)

    if args.use_cuda and not torch.cuda.is_available():
        print("=> There is no cuda device!!!!")
        device = "cpu"
    else:
        device = torch.device("cuda" if args.use_cuda else "cpu")

    kwargs = {"pin_memory": True} if args.use_cuda else {}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = get_datacenter_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = MSELoss().to(device)

    metrics = {
        "mae": mean_absolute_error,
        "rmse": root_mean_squared_error,
    }

    server_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    server = TorchServer(server_opt)

    trainer = ParallelTrainer(
        server=server,
        aggregator=_get_aggregator(),
        pre_batch_hooks=[],
        post_batch_hooks=[],
        max_batches_per_epoch=args.max_batches_per_epoch,
        log_interval=args.log_interval,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
    )

    test_loader = datacenter(
        data_dir=DATA_DIR,
        file_ids=DATACENTER_FILE_IDS,
        train=False,
        batch_size=args.test_batch_size,
        shuffle=False,
        split_ratio=args.split_ratio,
        seed=args.seed,
        drop_last=False,
        **kwargs,
    )

    evaluator = DistributedEvaluator(
        model=model,
        data_loader=test_loader,
        loss_func=loss_func,
        device=device,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
    )

    for worker_rank in range(N_WORKERS):
        worker = initialize_worker(
            trainer=trainer,
            worker_rank=worker_rank,
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            device=device,
            kwargs={},
        )
        trainer.add_worker(worker)

    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        evaluator.evaluate(epoch)


if __name__ == "__main__":
    main(args)
