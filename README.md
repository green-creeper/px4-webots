# px4-webots
Attempt to run webots sim with PX4

> **NOTE** this is the very beginning of the project where I try to adapt ardupilot webots simulation to PX4 stack
> Currently it has problems with sensor data and crashes after take off which I'm trying to solve investigating the proper way of sendor mapping.
> The project is draft and made for personal use, I'm sharing it as an example, though any suggestions and improvements are welcomed.

## How to run

- Follow PX4 development setup [guidelines](https://docs.px4.io/main/en/dev_setup/dev_env.html) to install PX4 development toolkit

```
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot/Tools/setup
sh ubuntu.sh
```


- Start SITL simulation
```
cd <PX4 root>
make px4_sitl none_iris
```

- Start webots (Iris.wbt for now)

- Start qGroundControl



Sitl should always be started befor webots as webots controller reads initial messages and starts iterating only after receiving them.
After each crush both webots and SITL should be restarted.