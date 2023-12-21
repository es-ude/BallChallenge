# BallChallenge

## Installation Instructions

### Project

Make sure that you have `python3.11` and `git` installed on your system. To install `git` you can follow the installation instructions for your system as described [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

An easy and platform independent method to make sure you have a working `python3.11` environment, that can also exists in parallel to other python versions is to use Miniconda (installation instruction for Windows, macOS and Linux can be found [here](https://docs.conda.io/projects/miniconda/en/latest/)).

If you have Miniconda installed you can create a `python3.11` environment for this project using the following command:

```bash
conda create -n ballchallenge python=3.11
```

To activate this environment use the following command:

```bash
conda activate ballchallenge
```

After activating the ballchallenge conda environment or any other `python3.11` environment you can download the project and install all necessary dependencies using the following commands:

```bash
git clone https://github.com/es-ude/BallChallenge.git
cd BallChallenge/

# TODO: Decide if this should be merged to main or not
git checkout explore-data-and-adapt-model

pip install -r requirements.txt
```

### IDE

As an IDE for this project we recommend to use Visual Studio Code (installation instructions can be found [here](https://code.visualstudio.com/)) with the Python and Jupyter extension installed. But you can use any other IDE if you want (e.g. PyCharm).
