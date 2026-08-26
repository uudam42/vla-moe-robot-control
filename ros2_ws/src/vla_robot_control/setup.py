from setuptools import find_packages, setup

package_name = "vla_robot_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/mujoco_vla.launch.py"]),
        ("share/" + package_name + "/config", ["config/default_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="uudam",
    maintainer_email="agudanmu@gmail.com",
    description="Step 9 ROS2 deployment: MuJoCo bridge + VLA policy nodes over a RobotBackend abstraction.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mujoco_bridge_node = vla_robot_control.mujoco_bridge_node:main",
            "vla_policy_node = vla_robot_control.vla_policy_node:main",
        ],
    },
)
