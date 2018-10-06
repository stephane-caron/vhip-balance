# 3D Balance using both ZMP and COM height variations

Source code for https://hal.archives-ouvertes.fr/hal-01590509/document

## Installation

The following instructions were verified on various machines running Ubuntu
14.04 and 16.04.

- Install OpenRAVE: here are [instructions for Ubuntu 14.04](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html) as well as [for Ubuntu 16.04](https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html)
- Install Python and related dependencies: ``sudo apt-get install cython python python-dev python-pip python-scipy``
- Install Python packages: ``sudo pip install pycddlib quadprog``
- Install [CasADi](http://casadi.org). Pre-compiled binaries are available, but I recommend you [build it from source](https://github.com/casadi/casadi/wiki/InstallationLinux). When installing IPOPT, make sure to install the MA27 linear solver (``ThirdParty/HSL`` folder).

You can then clone the repository and its submodule via:

```bash
git clone --recursive https://github.com/stephane-caron/3d-balance.git
```

## Usage

Run the main script `stepping.py`, then call one of the two main functions:

- `edit()` to switch to editor mode (moving the contact around, changing initial velocity, etc.)
- `go()` to execute the motion from the current state.

You can also checkout the benchmarking script in the `benchmark/` sub-folder.
It was used to measure computation times reported in Section V of the paper.

Due to copyright, we cannot release the COLLADA model ``HRP4R.dae`` used to
produce the accompanying video and paper illustrations. It is replaced by
[JVRC-1](https://github.com/stephane-caron/openrave_models/tree/master/JVRC-1),
which has the same kinematic chain.
