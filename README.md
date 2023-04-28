> bipyt semestral version

---

# bvviz - The simulation and visualization of Bernstein-Vazirani quantum protocol

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/chutommy/bvviz/blob/main/LICENSE)

**bvviz** is a web app providing a user-friendly playground for running noisy quantum simulations
and visualizing the Bernstein-Vazirani quantum algorithm!

![img.png](assets/images/screenshot.png)

## Installation

To install **bvviz**, first download the source code to your local machine.

```bash
git clone https://github.com/chutommy/bvviz.git
cd chutommy/
```

Use the open-source package manager [conda](https://www.anaconda.com/) to install all dependencies.

```bash
conda env create -f environment.yml -n bvviz
```

## Usage

Run `make` command to start the `bvviz` server.

```bash
make run
```

Now feel free to launch to your favorite browser of choice and go to
[localhost:8080](http://localhost:8080/).

![img.png](assets/images/screenshot2.png)

The moment you open a new session, a default experiment is performed.
At the top of the web page there's a short introduction to the **bvviz** project. You can expand
the _About_ section to get to know the Bernstein Vazirani problem.

Below the introduction the fun begins. All the metrics, plots, charts and stats are thoroughly
described. In order to display the tooltip, hover over the question mark.

![img.png](assets/images/screenshot4.png)

If the statistics and visualizations are inadequate for you, at the bottom of the page there
are multiple options to download the results of the experiment:

1. **OpenQASM** is the quantum circuit of the experiment.
2. **Counts** is the number of measurements for each value.
3. **Measurements** contains the individual measurements in the order they were captured.

### Custom configurations

Finally, let's take a look at the configuration sidebar.

In the configuration you are free to customize the device of the simulation, including the
backend system, number of shots, your own secret string to the Bernstein-Vazirani problem,
and your own noise and transpiler model!

I recommend secret strings of size 4-12. If you want to experiment with bigger secret strings
I advise to not go over 18 since the computational demand of each experiment grows exponentially.
Simulating a Bernstein-Vazirani protocol with a secret string of size 15 already takes a few minutes.

*Please avoid using Prague backend. It's the only IBM's 'fake' system that's broken right now.
Thank you.*

![img.png](assets/images/screenshot3.png)

## Tests

To test that all parts of the code are working properly, run the respective `make` command.

```bash
make test-unit
```

To run UI tests, make sure the server is online (on port 8080). The testing suit will create a new
session for testing purpose, so you don't need to worry about your current experiment.

Shortly after starting the test, a new testing browser will be opened. Please don't intervene
the testing while it's running (ideally sit back and don't interact with the interface at all).

```bash
# skip if server is already running
make run

# open a new terminal
make test-ui
```

> The app is made using [streamlit](https://streamlit.io/) library which currently does not support
> generating unique HTML id's nor classes. This means the UI testing is made by selecting
> by-product's of the HTML generated content. This workaround isn't reliable however right now
> there are no other relevant ways of testing the UI. These issues were/are actively being discussed
> on multiple GitHub issue pages:
>
> 1. https://github.com/streamlit/streamlit/issues/3888
> 2. https://github.com/streamlit/streamlit/issues/5437
> 3. https://github.com/streamlit/streamlit/issues/6482
>
> Proper UI testing will be implemented once streamlit delivers a way to uniquely identify HTML
> elements.