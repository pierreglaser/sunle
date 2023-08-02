In this experiment, we perform inference using SUNLE on a model of the Pyloric network in the crab cancer borealis [1].
We compare SUNLE to two methods already used for inference for this model: SNVI [2], and TSNPE [3]

[1] Recordings from the c. borealis stomatogastric nervous system at different temperatures in the decentralized condition
    Haddad, Sara Ann and Marder, Eve

[2] Variational methods for simulation-based inference
    Manuel Glöckler, Michael Deistler, Jakob H. Macke

[3] Truncated proposals for scalable and hassle-free simulation-based inference
    Michael Deistler, Pedro J Goncalves, Jakob H Macke


### Running SUNLE
To run the SUNLE experiment, simply activate the UNLE virtual environment and run the `experiments.ipynb`
notebook (run `jupyter lab`, open the notebook, and run it).

### Running TSNPE
To run the TSNPE experiments, from the current folder, run the following commands:

```sh
cd ../../src/tsnpe;
mamba env create -f environment_vm.yml:
mamba activate tsnpe_neurips;
python -m pip install -e ./l5pc;
python -m pip install -e ./sbi;
python -m pip install -e ../pyloric;
cd ./l5pc;
source run_exps.sh
```

### Running SNVI

TODO: Add instructions for running SNVI (we use a fork that allows to collect metrics about each round).
The results we obtain align with the results reported in the paper.
