clear all; clc;
global sys_interval sample_interval Q R;
global delta_umax ;
global model_name data_path timetoc simTime tempQ RTimes;       
global seq_nearest_idx start_idx;
global domain_name ref_dataset_name alg_name root_path;
global Np Nc;  
domain_name = "DClassSUV";
alg_name = "MPC";
sys_interval = 50;  % sample interval equals 50ms
sample_interval = ceil(sys_interval / 10);
ref_dataset_name = "DClassSUVDatasets";
Np = 30; % prediction horizon
Nc = 30; % control horizon
% ***************** must modify the current path **********
root_path = "F:/XYQ/DDK_open_access";       
preprocess_data_path = sprintf("%s/CarsimDClassSUVDatasets", root_path);
addpath(preprocess_data_path)
train_file_num = 30; val_file_num = 4; test_file_num = 4;

start_idx = 2; %19582;
seq_nearest_idx = [];
simTime = 1000;
model_name = 'DDK_DCSUV_50ms';      % This is the name of the simulink file
delta_umax = [0.5; 0.2]; % 
datatype = 'all'; % {'all', 'train', 'val', 'test'}
is_save = true;
switch datatype
    case 'all'
        file_num = train_file_num + val_file_num + test_file_num;
        s_index = 1;
    case 'train'
        file_num = train_file_num;
        s_index = 1;
    case 'val'
        file_num = val_file_num;
        s_index = train_file_num + 1;
    case 'test'
        file_num = test_file_num;
        s_index = train_file_num + 1 + val_file_num;
end
tempQ = diag([20, 1000, 1000, 1000, 20, 20]);    % matrix Q
RTimes = diag([5, 10000]); % matrix R

for i=s_index:s_index+file_num
    seq_nearest_idx = [];
    if i <= train_file_num
        datatype = 'train';
        idx = i - 1;
    elseif i > train_file_num && i <= train_file_num + val_file_num
        datatype = 'val';
        idx = i - 1 - train_file_num;
    elseif i > train_file_num + val_file_num && i <= train_file_num + val_file_num + test_file_num
        datatype = 'test';
        idx = i - 1 - train_file_num - val_file_num;
    end

    if strcmp(datatype, 'train')
        data_path = sprintf(strcat(preprocess_data_path, "/Carsim_DClassSUV_%d.mat"), idx);
    elseif strcmp(datatype, 'test')
        data_path = sprintf(strcat(preprocess_data_path, "/Carsim_DClassSUV_test_%d.mat"), idx);
    elseif strcmp(datatype, 'val')
        data_path = sprintf(strcat(preprocess_data_path, "/Carsim_DClassSUV_val_%d.mat"), idx);
    end
    ref_data = load(data_path);
    sim_time = floor(size(ref_data.Pos, 1) / (1000 / sys_interval)); 
    sim_time = min(sim_time - 2, simTime);
    timetoc = zeros(sim_time * 100 + 1, test_file_num);
    time = int2str(sim_time);
    out = sim(model_name, 'StopTime',time, 'SaveState', 'on', 'SaveOutput', 'on');
    x_pred = out.vehicle_data;
    x_pred = x_pred.data;
    x_pred = x_pred(:, 2:7);
    x_pred = x_pred(1:sample_interval:end, :);
    mpc_data = out.control;
    mpc_data = mpc_data.data;
    if size(mpc_data, 2) == 2
        zero_engine = zeros(size( mpc_data, 1), 2);
        mpc_data = [mpc_data, zero_engine];
    end
    
    if ref_dataset_name == "DClassSUVDatasets"
        X_max = [20, 0.5, 0.5];
        X_min = [0, -0.5, -0.5];
    end
    ref_data = load(data_path);
    ref_pos = ref_data.Pos;
    ref_x = ref_data.X;
    % anti-normalization to reference
    ref_x = (ref_x + 1) .* (X_max - X_min) / 2 + X_min;
    ref_x = [ref_pos, ref_x];
    plot_result(x_pred, ref_x, mpc_data, datatype, idx, is_save)
    close();
end
