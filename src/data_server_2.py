from syft.workers.websocket_server import WebsocketServerWorker
import torch
import syft as sy

hook = sy.TorchHook(torch)
server = WebsocketServerWorker(
    id="b",
    host="0.0.0.0",
    port="8778",
    hook=hook
)

dataset = sy.BaseDataset(
    data=torch.tensor([[1,0.],[1,1.]], requires_grad=True), 
    targets=torch.tensor([[1],[1.]], requires_grad=True)
)

server.add_dataset(dataset, key="toy")
server.start()