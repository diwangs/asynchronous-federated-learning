import asyncio
import torch
import syft as sy
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker
import torch.nn.functional as F

# Loss function
@torch.jit.script
def loss_fn(pred, target):
    return F.mse_loss(input=pred, target=target)

# A Toy Model
model = torch.jit.trace(torch.nn.Linear(2,1), torch.zeros([2], dtype=torch.float))

async def train_loop(client):
    global model

    for _ in range(100):
        train_config = sy.TrainConfig(
            model=model,
            loss_fn=loss_fn,
            batch_size=1,
            shuffle=True,
            max_nr_batches=1,
            epochs=1,
            optimizer="Adam",
        )
        train_config.send(client)
        loss = await client.async_fit(dataset_key="toy", return_ids=[0])
        new_model = train_config.model_ptr.get().obj

        # model += 1/n * (new_model - model), the stateful way
        grad = utils.add_model(new_model, utils.scale_model(model, -1))
        model = utils.add_model(utils.scale_model(model, -1), utils.scale_model(grad, 0.5))
        
        print(client.id, loss.data)

async def main():
    hook = sy.TorchHook(torch)
    kwargs_websocket = {"hook": hook, "host": "0.0.0.0", "verbose": True}
    a = WebsocketClientWorker(id="a", port=8777, **kwargs_websocket)
    b = WebsocketClientWorker(id="b", port=8778, **kwargs_websocket)
    await asyncio.gather(*[train_loop(x) for x in [a, b]]) # hacky...

asyncio.run(main())

print(model(torch.tensor([[0, 0.]])))