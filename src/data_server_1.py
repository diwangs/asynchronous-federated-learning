from syft.workers.websocket_server import WebsocketServerWorker
import torch
import syft as sy

hook = sy.TorchHook(torch)
server = WebsocketServerWorker(
    id="a",
    host="0.0.0.0",
    port="8777",
    hook=hook
)

dataset_a = sy.BaseDataset(
    data=torch.tensor([[0,0.],[0,1.]], requires_grad=True), 
    targets=torch.tensor([[0],[0.]], requires_grad=True)
)

server.add_dataset(dataset_a, key="toy")
server.start()