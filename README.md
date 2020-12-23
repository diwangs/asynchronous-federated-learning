# Async FL using PySyft
My bachelor thesis project, to fulfill a degree requirement from Bandung Institute of Technology

by Senapati Sang Diwangkara, 13516107

This [paper](paper.pdf) was published on [ICITSI 2020](https://ieeexplore.ieee.org/document/9264958)

This work examines the effect of asynchronicity in aggregation algorithm (i.e. when the nodes' training round / epoch don't have to be in sync with one another). 
In particular, I'm interested in the effect of data imbalance. 
Intuitively speaking, if we have some true/false dataset that is split between 2 nodes, but each node only have either true or false data, and one node is faster than the other, than the trained model should perform worse than if the model is trained centrally, right?
So here, we tried to verify that intuition quntitatively.

## Experiment Architecture
![Arch](arch.png)

Clients are implemented in `src/client.py`. 
It will spool up some client threads to connect to each server, and an evaluator process to evaluate the latest model periodically. 
Each client threads will communicate with the evaluator process with the `evaluator_q` variable, that acts as a queue. 
The threads will enqueue their evaluation order and they will be consumed by the evaluator process, which then will evaluate the snapshoted model and print the result.

Servers are implemented in `src/servers.py`. 
It will split a given dataset (using `src/split_dataset.py`) and started the servers that will host said data.

To measure optimization performance, we use `yappi` as a profiler to measure how much time each function is spending.

## Questions?
Poke me on [Twitter!](twitter.com/diwangs_)