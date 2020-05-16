import sys
import time
import asyncio
from threading import Thread, Lock
from multiprocessing import Process
from copy import deepcopy
from datetime import datetime
import logging
logging.basicConfig()
logger = logging.getLogger("client")
logger.setLevel(logging.DEBUG)

import torch
from torchvision import datasets, transforms
import syft as sy
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker
from sklearn.metrics import f1_score

from base_model import Net, loss_fn

# Initialize global variable
model = torch.jit.trace(Net(), torch.zeros([1, 1, 28, 28], dtype=torch.float))
lock = Lock()

epochs = {}
last_smallest_epoch = 0

eval_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
eval_dataset_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=100, shuffle=False)

def main(n_server, staleness_threshold):
    threads = []
    stop_flag = False
    try:
        hook = sy.TorchHook(torch)
        
        # Connect to servers, keep trying indefinetly if failed
        kwargs_websocket = {"hook": hook, "host": "0.0.0.0", "verbose": True}
        servers = []
        for i in range(n_server):
            while True:
                try:
                    servers.append(WebsocketClientWorker(id=f"dataserver-{i}", port=8777+i, **kwargs_websocket))
                    # TODO: reconnect
                    break
                except ConnectionRefusedError:
                    continue

        logger.info("Training starts!")
        threads = [Thread(target=train_loop, args=(server, staleness_threshold, lambda: stop_flag)) for server in servers]
        # threads.append(Thread(target=evaluator_thread, args=(lambda: stop_flag, )))
        for thread in threads:
            thread.start()
        # The original thread becomes evaluator
        start_time = datetime.now()
        while all([thread.is_alive() for thread in threads]):
            p = Process(target=evaluate, args=(model, eval_dataset_loader, start_time))
            p.start()
            p.join()
            time.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        logger.info("\nGracefully shutting client down...")
        stop_flag = True
    finally:
        for thread in threads:
            thread.join()
        p.terminate()
        p.join()

def train_loop(server, staleness_threshold, should_stop):
    global epochs
    global last_smallest_epoch

    for epoch in range(1000):
        # Check staleness here
        if epochs:
            while(epoch - 1 - min(epochs.values()) > staleness_threshold):
                if staleness_threshold != 0:
                    print(f"{server.id} is at {epoch}, while min epoch is at {min(epochs.values())}")
                if should_stop():
                    break
                time.sleep(1)

        if should_stop():
            break

        loss = train(server)
        logger.debug(f"{server.id} {epoch} {loss}")

        # Update global state
        epochs[server.id] = epoch
        smallest_epoch = min(epochs.values())
        last_smallest_epoch = smallest_epoch

    server.close()

def train(server):
    global model
    global lock

    old_model = utils.add_model(utils.scale_model(Net(), 0), model) # clone model

    train_config = sy.TrainConfig(
        model=model,
        loss_fn=loss_fn,
        epochs=1,
        batch_size=32,
        max_nr_batches=1, # report back after a batch is complete
        shuffle=True,
        optimizer="SGD", # Adam doesn't work properly because momentum doesn't work?
        optimizer_args={"lr":0.1} 
    )
    train_config.send(server)

    loss = server.fit(dataset_key="mnist", return_ids=[0])
    new_model = train_config.get_model().obj

    # Asynchronous federated averaging: model += 1/n * (new_model - old_model), the stateful way
    weight = 1
    grad = utils.add_model(new_model, utils.scale_model(old_model, -1))
    scaled_grad = utils.scale_model(grad, weight)
    with lock:
        model = utils.add_model(model, scaled_grad)

    return loss.data

def evaluate(model, eval_dataset_loader, start_time):
    dur = datetime.now() - start_time
    
    y_pred = []
    for data, _ in eval_dataset_loader:
        y_pred.extend(model(data).detach().numpy().argmax(axis=1))
    f1 = f1_score(eval_dataset.targets, y_pred, average='micro')

    logger.info(f"{dur}, {f1}")

if __name__ == "__main__":
    try:
        N_SERVER = int(sys.argv[1])
        STALENESS_THRESHOLD = int(sys.argv[2])
        logger.debug(f"Will start client (model owner) that will connect to {N_SERVER} server(s)")
    except Exception as e:
        logger.error(e)
        sys.exit()
    
    main(N_SERVER, STALENESS_THRESHOLD)