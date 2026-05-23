from setuptools import find_packages, setup

package_name = 'turtle_draw_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='lorena',
    maintainer_email='lorena.garcia@sou.inteli.edu.br',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'draw_node = turtle_draw_pkg.draw_node:main',
        ],
    },
)
