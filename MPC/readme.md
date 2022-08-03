1. If you have installed CarSim 2019, and have imported the vehicle config from the '.cpar' file. Then modify the `root_path` in `DDK_mpc_main_DCSUV_50ms.m` as the current root path.
2. Run `DDK_mpc_main_DCSUV_50ms.m`. All the tracjectories in the training/testing/validating datasets will be tracked by DDK-MPC

## Note

1. `CarsimDClassSUVDatasets/` -- is the datasets in the paper, the traning/testing/validating files were randomly selected. Original files are in the folder `DDK/datasets`
2. `params_for_matlab_CarsimDClassSUV_2022_07_24_15_23.mat` -- is the params of the DDK model including weights and biases of the encoder, decoder, koopman operator.
