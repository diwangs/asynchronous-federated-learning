import sys
from syft.workers.websocket_server import WebsocketServerWorker
import syft as sy
from collections import Counter
from multiprocessing import Process
from time import sleep

from torchvision import datasets, transforms
from split_dataset import split_dataset_indices

from custom_server import CustomWebsocketServerWorker

# Argument parsing
try:
    N_SERVER = int(sys.argv[1])
    STDEV = int(sys.argv[2])
    print(f"Will start {N_SERVER} servers (data owners) with stdev of {STDEV}")
except Exception as e:
    print(e)
    sys.exit()

def run_data_server(id, indices):
    import torch # Each process should import torch to allow parallelization?

    try:
        hook = sy.TorchHook(torch)
        server = CustomWebsocketServerWorker(
            id=f"dataserver-{id}",
            host="0.0.0.0",
            port=f"{8777 + id}",
            hook=hook
        )

        mnist = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        is_kept_mask = torch.tensor([x in indices for x in range(len(mnist.targets))])

        dataset = sy.BaseDataset(
            data=torch.masked_select(mnist.data.transpose(0, 2), is_kept_mask)
                .view(28, 28, -1)
                .transpose(2, 0),
            targets=torch.masked_select(mnist.targets, is_kept_mask),
            transform=mnist.transform
        )

        server.add_dataset(dataset, key="mnist")
        print(f"Server {id} started")
        server.start()
    except (KeyboardInterrupt, SystemExit):
        print(f"Server {id} stoped")

if __name__ == "__main__":
    processes = []
    print("Splitting dataset...")
    targets = datasets.MNIST(
        root="./data",
        train=True,
        download=True
    ).targets.numpy()

    indices_per_class = [[] for _ in range(10)]
    for idx, value in enumerate(targets):
        indices_per_class[value].append(idx)

    indices_per_server = split_dataset_indices(indices_per_class, N_SERVER, STDEV)

    print("Starting servers...")
    for i in range(N_SERVER):
        processes.append(Process(target=run_data_server, args=(i, indices_per_server[i])))
    for process in processes:
        process.start()
    for process in processes:
        process.join()