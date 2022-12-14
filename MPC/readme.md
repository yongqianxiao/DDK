1. If you have installed CarSim 2019, and have imported the vehicle config from the '.cpar' file. Then modify the `root_path` in `DDK_mpc_main_DCSUV_50ms.m` as the current root path.
2. Run `DDK_mpc_main_DCSUV_50ms.m`. All the tracjectories in the training/testing/validating datasets will be tracked by DDK-MPC

## Note

1. `CarsimDClassSUVDatasets/` -- is the datasets in the paper, the traning/testing/validating files were randomly selected. Note that the longitudinal and lateral velocities, and the yaw rate of trajectories in datasets in this folder have been normalized. Original files are in the folder `DDK/datasets`. 
2. `params_for_matlab_CarsimDClassSUV_2022_07_24_15_23.mat` -- is the params of the DDK model including weights and biases of the encoder, decoder, koopman operator.
3. `DDK_DCSUV_50ms.slx` -- is the simulink file (MATLAB 2021b).
4.  `DDK_DCSUV_50ms_2020a.slx` -- simulink file (MATLAB 2020a).
