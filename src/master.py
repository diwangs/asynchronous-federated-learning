import asyncio
import torch
import syft as sy
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker
from torch import nn
import torch.nn.functional as F

N_SERVER = 10

# Loss function
@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)


# A Toy Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = torch.jit.trace(Net(), torch.zeros([1, 1, 28, 28], dtype=torch.float))

async def train_loop(server):
    global model
    epoch = 0

    for _ in range(100):
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

        # model += 1/n * (new_model - model), the stateful way
        grad = utils.add_model(new_model, utils.scale_model(model, -1))
        model = utils.add_model(utils.scale_model(model, -1), utils.scale_model(grad, 0.5))

        print(server.id, epoch, loss.data)
        epoch += 1

async def main():
    hook = sy.TorchHook(torch)
    kwargs_websocket = {"hook": hook, "host": "0.0.0.0", "verbose": True}
    servers = []
    for i in range(N_SERVER):
        servers.append(WebsocketClientWorker(id=f"dataserver-{i}", port=8777+i, **kwargs_websocket))
    await asyncio.gather(*[train_loop(server) for server in servers]) # hacky...

asyncio.run(main())