TODO:
* Parameterize data server with real data
* Create script to split data into n bin with incomplete classes
* Learn netem: simulate network loss and delay 
sudo tc qdisc del dev lo root

# Distributed Model Consistency Spectrum
* Synchronous
  * Models are combined at training time, 
* Stale-synchronous
* Asynchronous
* Model averaging
  * Models are combined post-training
* Ensemble
  * Models are never combined

# What to test
General theme: performance (convergence time) gain, so graph the accuracy vs time and epoch, and number of node waiting
* Benchmark: 
  * Centralized
  * Distributed (no loss and delay, evenly distributed, small number of node?)
    * 2, 5, 10, 15 nodes
* Staleness testing: 
  * Constant number of node? -> 10? -> mau diganti
  * Constant and uniform loss and delay (+ jitter and additional processing time?)
  * Variable standard deviation
  * Variable staleness threshold

* Things to improve: if some particular nodes are constantly slow, they bog everyone else
  * Test with different delay on some particular node?
* Asynchronous averaging weighting?

Question: 
* Node count constant aja? -> bisa jadi pengaruh
* Loss and delay uniform? Atau dibedain per particular node? Dan berdasarkan apa? How to formalize which node get how much delay?
  * Node banyak delay banyak - Node banyak delay dikit. For 0-std: random
  * Gausah...

Hypothesis: 
* Asynchronicity helps performance with the right level in the right condition
* This condition is partly determined by the "wildness" of the environment: 
  * non-IID-ness, unevenness of the data
  * Network quality of the servers
* Applying high asynchronicity in a certain wild env can result in model being inaccurate: slow but knowledgable server will lose the gradient "tug-of-war"
* Applying low asynchronicity in some other wild env can result in slowness: slow and low-information server can be left behind
* In a tame environment, there's a performance plateau anyway due to amdahl's law? just like regular distributed learning