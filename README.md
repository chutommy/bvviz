# bvviz - The Simulation and Visualization of the Bernstein-Vazirani Quantum Protocol

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/chutommy/bvviz/blob/main/LICENSE)

**bvviz** is an environment which provides a user-friendly playground for running noisy quantum simulations and visualizing the Bernstein-Vazirani quantum algorithm.

![img.png](assets/images/img1.png)

## Installation

To install **bvviz**, first download the source code to your local machine.

```bash
git clone https://github.com/chutommy/bvviz.git
cd bvviz/
```

Use the open-source package manager [conda](https://www.anaconda.com/) to install all dependencies to a new environment.

```bash
conda env create -f condaenv/environment.yml -n bvviz -q
```

## Usage

To perform simulations, activate the bvviz environment and initiate the server.

```bash
conda activate bvviz
make run
```

Now, launch a web browser and navigate to [localhost:8501](localhost:8501).

### Custom configurations

In the configuration, you are free to customize the device of the simulation as you want. However, for best results it is recommended to use secret strings with a length of 4-12. If you wish to experiment with larger secret strings, keep in mind that the computational demand grows exponentially with increasing size.

![img.png](assets/images/img2.png)

## Tests

To test that all parts of the code are working properly, run the respective `make` command.

```bash
make test-unit
```

To run UI tests, make sure the server is online (on port 8501). The testing suite will create a new session for testing purposes.

Shortly after initializating the test sequence, a new testing browser will be opened. Please do not intervene while it's running.

```bash
# skip if the server is already running
make run

# run in a new terminal
make test-ui
```
