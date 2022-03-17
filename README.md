# PySMCSampler

An SMC Sampler that interfaces to Stan via [PyBindStan](https://github.com/mjcarter95/PyBindStan). A template for sampling from a Stan model is provided in `main.py`. Currently, we use a random walk proposal and set L=q in the weight updates.

Please email any questions to m.j.carter2@liverpool.ac.uk