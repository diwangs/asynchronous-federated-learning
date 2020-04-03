import sys
import asyncio
import torch
import syft as sy
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker
from torch import nn
import torch.nn.functional as F

from base_model import Net, loss_fn

# Argument parsing
try:
    N_SERVER = int(sys.argv[1])
    print(f"Will start client (model owner) that will connect to {N_SERVER} server(s)")
except Exception as e:
    print(e)
    sys.exit()

# Initialize global variable
model = torch.jit.trace(Net(), torch.zeros([1, 1, 28, 28], dtype=torch.float))

async def train_loop(server):
    global model

    for epoch in range(100):
        # Check staleness here
        train_config = sy.TrainConfig(
            model=model,
            loss_fn=loss_fn,
            batch_size=1,
            shuffle=True,
            max_nr_batches=1,
            epochs=1,
            optimizer="Adam",
        )
        train_config.send(server)
        loss = await server.async_fit(dataset_key="mnist", return_ids=[0])
        new_model = train_config.model_ptr.get().obj

        # Asynchronous federated averaging: model += 1/n * (new_model - model), the stateful way
        # TODO: experiment with weight? Should be just 1 because of adam?
        weight = 1
        grad = utils.add_model(new_model, utils.scale_model(model, -1))
        model = utils.add_model(utils.scale_model(model, -1), utils.scale_model(grad, weight))

        print(server.id, epoch, loss.data)

async def main():
    hook = sy.TorchHook(torch)
    
    # Connect to servers, keep trying indefinetly if failed
    kwargs_websocket = {"hook": hook, "host": "0.0.0.0", "verbose": True}
    servers = []
    for i in range(N_SERVER):
        while True:
            try:
                servers.append(WebsocketClientWorker(id=f"dataserver-{i}", port=8777+i, **kwargs_websocket))
            except ConnectionRefusedError:
                continue
            break

    print("Training starts!")
    await asyncio.gather(*[train_loop(server) for server in servers]) # hacky...

asyncio.run(main())