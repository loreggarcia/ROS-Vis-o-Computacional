import sys
if sys.prefix == '/home/lorena/.local/share/mamba/envs/ros2-humble':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/lorena/ponderada-ROS/install/turtle_draw_pkg'
