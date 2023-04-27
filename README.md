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

Use the open-source package manager [conda](https://www.anaconda.com/) to install all requirements
and dependencies from an `environment.yml` file. Please be patient, this may take a while.

```bash
conda env create -f environment.yml -n bvviz
```

## Usage

To start the `bvviz` server, simply run a single `make` command.

```bash
make run
```

Now feel free to launch to your favorite browser of choice and go to
[localhost:8080](http://localhost:8080/).

### The layout

![img.png](assets/images/screenshot2.png)

The layout is very intuitive. First thing of, let's ignore the configuration on the left hand side
for a moment and focus on the main page.

The moment you open a new session, a default experiment is performed. Don't worry! You
will be able to run your own experiments later on.

At the top of the web page there's a short introduction to the **bvviz** project. You can expand
the _About_ section to get to know the Bernstein Vazirani problem.

Below the introduction the fun begins. All the metrics, plots, charts and stats are thoroughly
described. If there happens to be something that's too challenging to grasp or not explained well
enough, please let me know.

![img.png](assets/images/screenshot4.png)

If the statistics and visualizations are inadequate for you, at the bottom of the page there
are multiple options to download the result of the experiment:

1. **OpenQASM** is the quantum circuit of the experiment.
2. **Counts** is the number of measurements for each value.
3. **Measurements** contains the individual measurements in the order they were taken.

### Custom configurations

Finally, let's take a look at the configuration sidebar.

In the configuration you are free to customize the device of the simulation, including the
backend system, number of shots, your own secret string to the Bernstein-Vazirani problem,
and your own noise and transpiler model!

Notice the quantum systems are named after cities! Yes, these are IBM's quantum providers. You
are (almost) free to choose any backend you want. Just remember that the secret string must be
compatible with the backend you choose.

I recommend secret string of size 4-12. If you want to experiment with bigger secret string
I advise to not go over 18 since the computational demand of each experiment grows exponentially.
Simulating a Bernstein-Vazirani protocol with a secret of size 15 already takes up to few minutes.

Rather than waiting all day for an experiment to finish, play with different backends, noise models
and transpiling methods. Notice how device maps and circuit layouts change based on the quantum
system you choose.

*Please avoid using Prague backend. It's the only IBM's broken 'fake' system right now. Thank you.*

![img.png](assets/images/screenshot3.png)

## Tests

To test that all parts of the code are working properly, run the respective `make` command.

```bash
make test-unit
```

To run UI test, make sure the server is online (on port 8080). The testing suit will create a new
session for testing purpose, your current experiment won't be overriden.

Shortly after starting the test, a new testig browser will be opened. Please don't intervene
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