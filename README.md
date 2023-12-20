# DDK
files for "DDK: A Deep Koopman Approach for Longitudinal and Lateral Control of Autonomous Ground Vehicles"


This work is an improvement from our previous work, and it was accepted by ICRA2023: 

@ARTICLE{xiao2023deep, author={Xiao, Yongqian and Zhang, Xinglong and Xu, Xin and Liu, Xueqing and Liu, Jiahang}, journal={IEEE Transactions on Intelligent Vehicles}, title={Deep Neural Networks With Koopman Operators for Modeling and Control of Autonomous Vehicles}, year={2023}, volume={8}, number={1}, pages={135-146}, doi={10.1109/TIV.2022.3180337}}

@inproceedings{xiao2023ddk,
  title={DDK: A Deep Koopman Approach for Longitudinal and Lateral Control of Autonomous Ground Vehicles},
  author={Xiao, Yongqian and Zhang, Xinglong and Xu, Xin and Lu, Yang and Lil, Junxiang},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={975--981},
  year={2023},
  organization={IEEE}
}

## dataset format

column 1: sample time: Note that the sample interval is 10ms while we collected the datasets, but it equals 50ms while we identify the vehicle dynamics in the simulation. You just need to sample a frame of data every 5 frames from the datasets.

column 2~4: The global X(m), Y(m), and Yaw(rad) angle.

column 5~7: The longitudinal velocity (m/s), lateral velocity(m/s), and yaw rate(rad/s).

column 28~30: The front wheel angle(rad), throttle, and brake.

## DClassSUV_from_laptop_20220628.cpar

This is the Carsim config file. You can recover the Carsim simulation environment via improting this file in Carsim.

## Experiment videos

(2023-06-15)
Because the training set data of the DDK vehicle model are all collected from urban roads, in order to verify the generalization performance of our method, we have recently added real-world experiments on off-road roads. 

![classic RNN](https://od.lk/s/ODFfNjc3MzAxNjhf/off_road_ref.png)

![classic RNN](https://od.lk/s/ODFfNjc3MzAxNDJf/DKMPC_track_result.png)

Experiment videos can be obtained at https://yongqianxiao.github.io/2022/09/12/Experiment-videos-of-DDK/
