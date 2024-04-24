# BallChallenge

## Installation Instructions

### Project


### IDE

As an IDE for this project, we recommend using Visual Studio Code (installation instructions can be found [here](https://code.visualstudio.com/)) with the Python and Jupyter extension installed. However, you can use any other IDE that supports Jupyter notebooks (e.g. PyCharm).


### Get Data
Download the data from [sciebo](https://udue.de/FEgzS)

Copy/Move the Data to the ./data folder 

On Linux and Mac-OS you can use something like ... (**YOU NEED TO ADAPT THIS**)
```bash
cp /path/to/Downloads/ path/to/Project/data
```

### Install Python with MiniConda
Make sure you have `python3.11` and `git` installed on your system. To install `git` you can follow the installation instructions for your system as described [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

An easy and platform-independent method to make sure you have a working `python3.11` environment that can coexists with other versions of python is to use Miniconda (installation instructions for Windows, macOS and Linux can be found [here](https://docs.conda.io/projects/miniconda/en/latest/)).

Once you have Miniconda installed you need to init it for your shell:
If your shell is bash use...
```bash
conda init bash
```
Now reopen your shell!
And use the following command to create a `python3.11` environment for this project:

```bash
conda create -n ballchallenge python=3.11
```

To activate this environment, use the following command:

```bash
conda activate ballchallenge
```

After activating the ballchallenge conda environment or any other `python3.11` environment, you can download the project and install all necessary dependencies using the following commands:

```bash
git clone https://github.com/es-ude/BallChallenge.git
cd BallChallenge/

pip install -r requirements.txt
```
