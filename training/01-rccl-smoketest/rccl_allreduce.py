# rccl_allreduce.py
import os
import torch
from datetime import timedelta
import torch.distributed as dist

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))

    x = torch.tensor([rank + 1.0], device=device)

    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    expected = world_size * (world_size + 1) / 2
    ok = torch.allclose(x, torch.tensor([expected], device=device))

    if rank == 0:
        print(f"[RANK 0] all_reduce SUM = {x.item():.0f}  (expected {expected:.0f})  -> {'OK' if ok else 'NG'}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

