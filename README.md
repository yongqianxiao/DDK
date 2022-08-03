# DDK
files for DDK

## dataset format

column 1: sample time: Note that the sample interval is 10ms while we collected the datasets, but it equals 50ms while we identify the vehicle dynamics in the simulation. You just need to sample a frame of data every 5 frames from the datasets.

column 2~4: The global X(m), Y(m), and Yaw(rad) angle.

column 5~7: The longitudinal velocity (m/s), lateral velocity(m/s), and yaw rate(rad/s).

column 28~30: The front wheel angle(rad), throttle, and brake.

## DClassSUV_from_laptop_20220628.cpar

This is the Carsim config file. You can recover the Carsim simulation environment via improting this file in Carsim.
