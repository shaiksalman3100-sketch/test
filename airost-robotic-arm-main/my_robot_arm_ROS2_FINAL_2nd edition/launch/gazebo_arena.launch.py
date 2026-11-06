import os
import re
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def create_qr_material_files(pkg_share):
    env_dir = os.path.join(pkg_share, 'env_assets')
    mat_dir = os.path.join(env_dir, 'materials')
    scripts_dir = os.path.join(mat_dir, 'scripts')
    textures_dir = os.path.join(mat_dir, 'textures')
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(textures_dir, exist_ok=True)

    for color in ['red', 'green', 'blue']:
        src_png = os.path.join(env_dir, f'qr_{color}.png')
        dst_png = os.path.join(textures_dir, f'qr_{color}.png')
        if os.path.exists(src_png):
            with open(src_png, 'rb') as fsrc, open(dst_png, 'wb') as fdst:
                fdst.write(fsrc.read())

    material_script = """
material QR/Red
{
  technique
  {
    pass
    {
      texture_unit
      {
        texture qr_red.png
        filtering none none none
      }
    }
  }
}
material QR/Green
{
  technique
  {
    pass
    {
      texture_unit
      {
        texture qr_green.png
        filtering none none none
      }
    }
  }
}
material QR/Blue
{
  technique
  {
    pass
    {
      texture_unit
      {
        texture qr_blue.png
        filtering none none none
      }
    }
  }
}
"""
    with open(os.path.join(scripts_dir, 'qr.material'), 'w') as f:
        f.write(material_script)
    return scripts_dir, textures_dir


def build_gazebo_urdf(pkg_share):
    urdf_path = os.path.join(pkg_share, 'urdf', 'my_robot_arm.urdf')
    controllers_yaml = os.path.join(pkg_share, 'config', 'controllers.yaml')
    with open(urdf_path, 'r', encoding='utf-8-sig') as f:
        urdf = f.read()

    urdf = re.sub(r'<\?xml[^>]*\?>', '<?xml version="1.0" ?>', urdf)
    urdf = re.sub(r'<!--.*?-->', '', urdf, flags=re.DOTALL)
    urdf = urdf.replace('package://my_robot_arm', f'file://{pkg_share}')

    gazebo_additions = f"""
  <link name="world"/>
  <joint name="world_fixed" type="fixed">
    <parent link="world"/>
    <child link="base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <ros2_control name="GazeboSystem" type="system">
    <hardware>
      <plugin>gazebo_ros2_control/GazeboSystem</plugin>
    </hardware>
    <joint name="joint_1">
      <command_interface name="position"/>
      <state_interface name="position"/><state_interface name="velocity"/>
    </joint>
    <joint name="joint_2">
      <command_interface name="position"/>
      <state_interface name="position"/><state_interface name="velocity"/>
    </joint>
    <joint name="joint_3">
      <command_interface name="position"/>
      <state_interface name="position"/><state_interface name="velocity"/>
    </joint>
    <joint name="joint_4">
      <command_interface name="position"/>
      <state_interface name="position"/><state_interface name="velocity"/>
    </joint>
    <joint name="joint_5">
      <command_interface name="position"/>
      <state_interface name="position"/><state_interface name="velocity"/>
    </joint>
    <joint name="joint_6">
      <command_interface name="position"/>
      <state_interface name="position"/><state_interface name="velocity"/>
    </joint>
  </ros2_control>

  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>{controllers_yaml}</parameters>
    </plugin>
  </gazebo>

  <gazebo reference="base_link"><material>Gazebo/DarkGrey</material></gazebo>
  <gazebo reference="link_1"><material>Gazebo/Orange</material></gazebo>
  <gazebo reference="link_2"><material>Gazebo/Grey</material></gazebo>
  <gazebo reference="link_3"><material>Gazebo/Orange</material></gazebo>
  <gazebo reference="link_4"><material>Gazebo/Grey</material></gazebo>
  <gazebo reference="link_5"><material>Gazebo/Orange</material></gazebo>
  <gazebo reference="link_6"><material>Gazebo/Red</material></gazebo>
</robot>
"""
    urdf = re.sub(r'</robot>\s*$', gazebo_additions, urdf)
    out_path = os.path.join(pkg_share, 'urdf', 'my_robot_arm_gazebo.urdf')
    with open(out_path, 'w') as f:
        f.write(urdf)
    return urdf, out_path


def make_cube_sdf(name, gazebo_color, pkg_share):
    cube_stl = os.path.join(pkg_share, 'env_assets', 'cube.STL')
    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <link name="link">
      <inertial>
        <mass>0.05</mass>
        <inertia>
          <ixx>0.00002</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.00002</iyy><iyz>0</iyz><izz>0.00002</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>0.05 0.05 0.05</size></box></geometry>
        <surface><friction><ode><mu>10.0</mu><mu2>10.0</mu2></ode></friction></surface>
      </collision>
      <visual name="visual">
        <pose>-0.025 -0.025 -0.025 0 0 0</pose>
        <geometry><mesh><uri>file://{cube_stl}</uri></mesh></geometry>
        <material><script><name>Gazebo/{gazebo_color}</name></script></material>
      </visual>
    </link>
  </model>
</sdf>"""


def make_bin_sdf(name, qr_mat, pkg_share, scripts_dir, textures_dir):
    bin_stl = os.path.join(pkg_share, 'env_assets', 'bin.STL')
    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="floor_col">
        <pose>0 0 0.0025 0 0 0</pose>
        <geometry><box><size>0.16 0.16 0.005</size></box></geometry>
      </collision>
      <!-- Genuine SolidWorks bin.STL mesh rotated from Y-up to Z-up -->
      <visual name="solidworks_bin_visual">
        <pose>-0.08 0.08 0 1.570796 0 0</pose>
        <geometry><mesh><uri>file://{bin_stl}</uri></mesh></geometry>
        <material><script><name>Gazebo/White</name></script></material>
      </visual>
      <visual name="qr_visual">
        <pose>0 0 0.012 0 0 0</pose>
        <geometry><box><size>0.11 0.11 0.001</size></box></geometry>
        <material>
          <script>
            <uri>file://{scripts_dir}</uri>
            <uri>file://{textures_dir}</uri>
            <name>{qr_mat}</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def make_camera_sdf():
    return """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="overhead_camera">
    <static>true</static>
    <pose>0.0 0.0 1.25 0 1.570796 0</pose>
    <link name="camera_link">
      <sensor type="camera" name="overhead_cam">
        <update_rate>20.0</update_rate>
        <camera name="head">
          <horizontal_fov>1.05</horizontal_fov>
          <image><width>1000</width><height>1000</height><format>R8G8B8</format></image>
          <clip><near>0.05</near><far>5.0</far></clip>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
          <ros><namespace>/vision</namespace></ros>
          <camera_name>overhead_camera</camera_name>
          <frame_name>camera_link</frame_name>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>"""


def generate_launch_description():
    subprocess.run(['killall', '-9', 'gzserver', 'gzclient'], stderr=subprocess.DEVNULL)
    pkg_share = get_package_share_directory('my_robot_arm')
    scripts_dir, textures_dir = create_qr_material_files(pkg_share)
    robot_desc, urdf_out = build_gazebo_urdf(pkg_share)

    sdf_dir = os.path.join(pkg_share, 'env_assets')
    world_file = os.path.join(sdf_dir, 'sorting_world.world')

    for cname, gcol in [('cube_red', 'Red'), ('cube_green', 'Green'), ('cube_blue', 'Blue')]:
        with open(os.path.join(sdf_dir, f'{cname}.sdf'), 'w') as f:
            f.write(make_cube_sdf(cname, gcol, pkg_share))

    for bname, qmat in [('bin_red', 'QR/Red'), ('bin_green', 'QR/Green'), ('bin_blue', 'QR/Blue')]:
        with open(os.path.join(sdf_dir, f'{bname}.sdf'), 'w') as f:
            f.write(make_bin_sdf(bname, qmat, pkg_share, scripts_dir, textures_dir))

    cam_sdf = os.path.join(sdf_dir, 'overhead_camera.sdf')
    with open(cam_sdf, 'w') as f:
        f.write(make_camera_sdf())

    return LaunchDescription([
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file, '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
            output='screen',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc, 'use_sim_time': True}],
        ),
        TimerAction(
            period=3.0,
            actions=[
                Node(package='gazebo_ros', executable='spawn_entity.py',
                     arguments=['-entity', 'my_robot_arm', '-file', urdf_out, '-x', '0.0', '-y', '0.0', '-z', '0.0'], output='screen'),
                Node(package='gazebo_ros', executable='spawn_entity.py',
                     arguments=['-entity', 'cube_red', '-file', os.path.join(sdf_dir, 'cube_red.sdf'), '-x', '0.30', '-y', '-0.12', '-z', '0.03']),
                Node(package='gazebo_ros', executable='spawn_entity.py',
                     arguments=['-entity', 'cube_green', '-file', os.path.join(sdf_dir, 'cube_green.sdf'), '-x', '0.30', '-y', '0.00', '-z', '0.03']),
                Node(package='gazebo_ros', executable='spawn_entity.py',
                     arguments=['-entity', 'cube_blue', '-file', os.path.join(sdf_dir, 'cube_blue.sdf'), '-x', '0.30', '-y', '0.12', '-z', '0.03']),
                Node(package='gazebo_ros', executable='spawn_entity.py',
                     arguments=['-entity', 'bin_red', '-file', os.path.join(sdf_dir, 'bin_red.sdf'), '-x', '0.00', '-y', '0.34', '-z', '0.0']),
                Node(package='gazebo_ros', executable='spawn_entity.py',
                     arguments=['-entity', 'bin_green', '-file', os.path.join(sdf_dir, 'bin_green.sdf'), '-x', '-0.30', '-y', '0.16', '-z', '0.0']),
                Node(package='gazebo_ros', executable='spawn_entity.py',
                     arguments=['-entity', 'bin_blue', '-file', os.path.join(sdf_dir, 'bin_blue.sdf'), '-x', '-0.30', '-y', '-0.16', '-z', '0.0']),
                Node(package='gazebo_ros', executable='spawn_entity.py',
                     arguments=['-entity', 'overhead_camera', '-file', cam_sdf, '-x', '0.0', '-y', '0.0', '-z', '0.0']),
            ],
        ),
        TimerAction(
            period=6.0,
            actions=[
                Node(package='controller_manager', executable='spawner', arguments=['joint_state_broadcaster'], output='screen'),
                Node(package='controller_manager', executable='spawner', arguments=['arm_controller'], output='screen'),
            ],
        ),
    ])
