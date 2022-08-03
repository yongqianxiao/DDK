function [sys,x0,str,ts,simStateCompliance] = Xms_DDK_MPC_Controller_DCSUV(t,x,u,flag)
%
% The following outlines the general structure of an S-function.
%
switch flag

  %%%%%%%%%%%%%%%%%%
  % Initialization %
  %%%%%%%%%%%%%%%%%%
  case 0
    [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes;

  %%%%%%%%%%%%%%% 这里不用！！！
  % Derivatives 该函数仅在连续系统中被调用，用于产生控制系统状态的导数%
  %%%%%%%%%%%%%%%
  case 1
    sys=mdlDerivatives(t,x,u);

  %%%%%%%%%%
  % Update 该函数仅在离散系统中被调用，用于产生控制系统的下一个状态 %
  %%%%%%%%%%
  case 2
    sys=mdlUpdate(t,x,u);

  %%%%%%%%%%%
  % Outputs %
  %%%%%%%%%%%
  case 3
    sys=mdlOutputs(t,x,u);

  %%%%%%%%%%%%%%%%%%%%%%%
  % GetTimeOfNextVarHit %
  %%%%%%%%%%%%%%%%%%%%%%%
  case 4
    sys=mdlGetTimeOfNextVarHit(t,x,u);

  %%%%%%%%%%%%%
  % Terminate 相当于构析函数，结束该仿真模块时被调用%
  %%%%%%%%%%%%%
  case 9
    sys=mdlTerminate(t,x,u);

  %%%%%%%%%%%%%%%%%%%%
  % Unexpected flags %
  %%%%%%%%%%%%%%%%%%%%
  otherwise
    DAStudio.error('Simulink:blocks:unhandledFlag', num2str(flag));

end

% end sfuntmpl

%
%=============================================================================
% mdlInitializeSizes
% Return the sizes, initial conditions, and sample times for the S-function.
%=============================================================================
%
function [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes

%
% call simsizes for a sizes structure, fill it in and convert it to a
% sizes array.
%
% Note that in this example, the values are hard coded.  This is not a
% recommended practice as the characteristics of the block are typically
% defined by the S-function parameters.
%
sizes = simsizes;

sizes.NumContStates  = 0;
sizes.NumDiscStates  = 4; %离散状态量个数
sizes.NumOutputs     = 4; %输出量个数
sizes.NumInputs      = 6; %输入量个数
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1;   % at least one sample time is needed

sys = simsizes(sizes);
x0  = [0;0;0;0];
global U a b;
U = [0;0];
global r r_u;
global ddk data_path;
global Np Nc Nx Nu is_ls;
global expe_date sys_interval root_path koopman_mode latent_mode domain_name;
global index ref_index nearest_idx start_idx cMode C oMode;

wpath = "params_for_matlab_CarsimDClassSUV_2022_07_24_15_23.mat";

ddk = DDK(wpath);

a = ddk.A';
b = ddk.B';
a = double(a);
b = double(b);

ref_data = load(data_path);
ref_pos = ref_data.Pos;
ref_x = ref_data.X;
r_u = ref_data.U;
r = [ref_pos, ref_x];                        % 13 x 1005
index = 1;
ref_index = index;
nearest_idx = 1;
%% 构建常数数组，以节省计算时间
% 构建权重矩阵Q 和 R
global Q R PHI THETA H A_l Row tempQ RTimes;
Row = 100; %松弛因子
Nx = ddk.space_dim;
Nu = ddk.u_dim;
if size(tempQ, 1) == 1 || size(tempQ, 2) == 1
    tempQ = diag(tempQ);
end
Q = kron(eye(Np), tempQ);

if length(RTimes) == 1
    R = eye(Nu*Nc)*RTimes;
elseif size(RTimes, 1) == 2 && size(RTimes, 2) == 1
    R = kron(eye(Nc), diag(RTimes));
elseif size(RTimes, 1) == 2 && size(RTimes, 2) == 2
    R = kron(eye(Nc), (RTimes));
end
% 构建以kesi为状态的，状态方程的A，B，C矩阵
A_cell = cell(2,2);
B_cell = cell(2,1);
A_cell{1,1} = a;
A_cell{1,2} = b;
A_cell{2,1} = zeros(Nu,Nx);
A_cell{2,2} = eye(Nu);
B_cell{1,1} = b;
B_cell{2,1} = eye(Nu);
A = cell2mat(A_cell);
B = cell2mat(B_cell);
% C = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0;];
C = [eye(ddk.s_dim) zeros(ddk.s_dim, ddk.space_dim - ddk.s_dim + Nu)];

% 构建PHI， THETA
PHI_cell = cell(Np, 1);
THETA_cell = cell(Np, Nc);
for j = 1:1:Np
  PHI_cell{j,1} = C*A^j;
  for k = 1:1:Nc
      if k <= j
          THETA_cell{j,k} = C*A^(j-k)*B;
      else
          THETA_cell{j, k} = zeros(ddk.s_dim, Nu);
      end
  end
end
PHI = cell2mat(PHI_cell); % (Nx*Np)*(Nx+Nu)
THETA = cell2mat(THETA_cell); % (Nx*Np)*(Nu*Nc)
% 构建二次规划问题的目标函数 矩阵H, f'   待求解变量(deltaU; 松弛变量)
H_cell = cell(2,2);
H_cell{1,1} = 2*THETA'*Q*THETA+R;
H_cell{1,2} = zeros(Nu*Nc,1);
H_cell{2,1} = zeros(1,Nu*Nc);
H_cell{2,2} = Row; % 权重系数
H = cell2mat(H_cell);
H = (H' + H) / 2;
% 构建A_1
A_t = zeros(Nc,Nc); %瑙乫alcone璁烘枃P181
for p = 1:1:Nc
  for q = 1:1:Nc
      if q<=p
          A_t(p,q) = 1;
      else
          A_t(p,q) = 0;
      end
  end
end
A_l = kron(A_t, eye(Nu));
%% 构建控制约束
global Umin Umax lb ub delta_umax;
umin = [-1; -1];
umax = [1; 1];
Umin = kron(ones(Nc, 1), umin);
Umax = kron(ones(Nc, 1), umax);
delta_umin = -delta_umax;   
delta_Umin = kron(ones(Nc, 1), delta_umin);
delta_Umax = kron(ones(Nc, 1), delta_umax);
M = 0;
lb = [delta_Umin; 0];
ub = [delta_Umax; M];
%
% str is always an empty matrix
%
str = [];
ts = [sys_interval/1000 0]; % sample time:[period, offset]

% Specify the block simStateCompliance. The allowed values are:
%    'UnknownSimState', < The default setting; warn and assume DefaultSimState
%    'DefaultSimState', < Same sim state as a built-in block
%    'HasNoSimState',   < No sim state
%    'DisallowSimState' < Error out when saving or restoring the model sim state
simStateCompliance = 'UnknownSimState';

% end mdlInitializeSizes

%
%=============================================================================
% mdlDerivatives
% Return the derivatives for the continuous states.
% 该函数仅在连续系统中被调用，用于产生控制系统状态的导数
%=============================================================================
%
function sys=mdlDerivatives(t,x,u)

sys = [];

% end mdlDerivatives

%
%=============================================================================
% mdlUpdate
% Handle discrete state updates, sample time hits, and major time step
% requirements.
% 该函数仅在离散系统中被调用，用于产生控制系统的下一个状态
%=============================================================================
%
function sys=mdlUpdate(t,x,u)

sys = x;

% end mdlUpdate

%
%=============================================================================
% mdlOutputs
% Return the block outputs.
%=============================================================================
%
function sys=mdlOutputs(t,x,u)
  global a b u_piao;
  global U;
  global kesi;
  global r ddk;
  global index nearest_idx seq_nearest_idx start_idx;
  global Np Nc Nx Nu timetoc;
  global Q PHI THETA H A_l;
  global Umin Umax lb ub;
  global model_name bad_count last_state;
  global sample_interval;
%   sample_interval = 1;
  index = index + 1;
  tic
  fprintf('Update start, t=%6.3f\n',t)
  %% 状态
  kesi = zeros(Nx+Nu, 1);
  u(3) = u(3) * pi / 180;   % yaw angle
  u(4) = u(4) / 3.6; u(5) = u(5) / 3.6; % U(4) u(5) are vx and vy
  u(6) = u(6)*pi/180; %Carsim 输入的是角度，这里转成弧度，u(6)为carsim输出的yaw rate
  x_cur = u';

    if index == 2 % initialize
        [start_idx, min_dis] = nearestPoint(r, x_cur, index, size(r, 1));
        nearest_idx = start_idx;
        bad_count = 0;
        last_state = x_cur;
    end
    if sum(abs(last_state - x_cur)) < 0.001
        bad_count = bad_count + 1;
    else
        bad_count = 0;
    end
    last_state = x_cur;
    [tmp_nearest_idx, min_dis] = nearestPoint(r, x_cur, nearest_idx, 500);
    nearest_idx = max(nearest_idx, tmp_nearest_idx);
    seq_nearest_idx = [seq_nearest_idx, nearest_idx];
    temp_refr = r(nearest_idx:sample_interval:Np*sample_interval+nearest_idx-1, :);
    if nearest_idx > size(r, 1) - Np * sample_interval - 200
        set_param(model_name, 'SimulationCommand', 'stop')
    end
    tmp_cur = [u' min_dis]
    if bad_count > 50
        set_param(model_name, 'SimulationCommand', 'stop')
    end
  ref_r = ddk.get_reference(temp_refr, temp_refr(1, 1:3));  % transform to local coordinate and normalize
    x_cur = ddk.normalization(x_cur, temp_refr(1, 1:3));    % transform to local coordinate and normalize
  x_cur = double(ddk.encoder(x_cur)');
  % ------------------- time-variant system, reinitilization -------------
  kesi(1:ddk.space_dim, :) = x_cur;
  kesi(end-1) = U(1);
  kesi(end) = U(2);
  
  %% 矩阵初始化
  u_piao = zeros(Nx, Nu);
  
  %% 构建二次规划问题的目标函数 矩阵H, f'   待求解变量(deltaU; 松弛变量)
temp_r = double(reshape(ref_r', [ddk.s_dim * Np, 1]));

  temp_cur = PHI * kesi;
  error = PHI*kesi - temp_r;
  f_cell = cell(1,2);
  f_cell{1,1} = 2*error'*Q*THETA;
  f_cell{1,2} = 0;
%  f = (cell2mat(f_cell))';
  f = cell2mat(f_cell);
  Ut = kron(ones(Nc, 1), U); % U 表示上一时刻的U，历史的U

  A_cons_cell = {A_l zeros(Nu*Nc,1); -A_l zeros(Nu*Nc,1)};
  b_cons_cell = {Umax-Ut; -Umin+Ut};
  A_cons = cell2mat(A_cons_cell);
  b_cons = cell2mat(b_cons_cell);
  
 %% solve QP
 options = optimset('Algorithm','interior-point-convex');
 options.MaxIter = 10000000000;
 [X, fval, exitflag] = quadprog(H,f',A_cons,b_cons,[],[],lb,ub,[],options);

 if isempty(X)
     u_piao(1) = 0;
     u_piao(2) = 0;
 else
     u_piao(1) = X(1);
     u_piao(2) = X(2);
 end
 U(1) = kesi(end-1)+ u_piao(1);
 U(2) = kesi(end)+ u_piao(2);
 u_real(1) = U(1) ;
 u_real(2) = U(2) ;
 steer_L1 = u_real(1) * ddk.a_max(1, 1) * 180 / pi;
 steer_R1 = steer_L1;
 throttle = 0; brake = 0;
 if u_real(2) >= 0
    throttle = u_real(2) *(ddk.a_max(1, 2) - ddk.a_min(1, 2)) + ddk.a_min(1, 2);
 else
     brake = u_real(2) * (ddk.a_max(1, 3) - ddk.a_min(1, 3)) + ddk.a_min(1, 3);
     brake = abs(brake);
 end
 output = [steer_L1; steer_R1; throttle; brake];
%  output = [steer_L1; steer_R1];
 sys = output;
 timetoc(index) = toc;
% End of mdlOutputs.

%
%=============================================================================
% mdlGetTimeOfNextVarHit
% Return the time of the next hit for this block.  Note that the result is
% absolute time.  Note that this function is only used when you specify a
% variable discrete-time sample time [-2 0] in the sample time array in
% mdlInitializeSizes.
%=============================================================================
%
function sys=mdlGetTimeOfNextVarHit(t,x,u)

sampleTime = 1;    %  Example, set the next hit to be one second later.
sys = t + sampleTime;

% end mdlGetTimeOfNextVarHit

%
%=============================================================================
% mdlTerminate
% Perform any end of simulation tasks.
%=============================================================================
%
function sys=mdlTerminate(t,x,u)

sys = [];

% end mdlTerminate
