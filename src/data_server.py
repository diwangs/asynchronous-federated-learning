from syft.workers.websocket_server import WebsocketServerWorker
import torch
import syft as sy
from collections import Counter
from multiprocessing import Process

from torchvision import datasets, transforms
from split_dataset import split_dataset_indices

N_SERVER = 10
STDEV = 0

def run_data_server(id, indices):
    hook = sy.TorchHook(torch)
    server = WebsocketServerWorker(
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

if __name__ == "__main__":
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
    processes = []
    for i in range(N_SERVER):
        processes.append(Process(target=run_data_server, args=(i, indices_per_server[i])))
    for process in processes:
        process.start()
    for process in processes:
        process.join()