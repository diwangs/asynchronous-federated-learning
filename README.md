TODO:
* Parameterize data server with real data
* Create script to split data into n bin with incomplete classes
* Learn netem: simulate network loss and delay 

* Python's asyncio is not the answer: we need to bridge different epoch -> different itteration in a for loop

# Distributed Model Consistency Spectrum
* Synchronous
  * Models are combined at training time, 
* Stale-synchronous
* Asynchronous
* Model averaging
  * Models are combined post-training
* Ensemble
  * Models are never combined