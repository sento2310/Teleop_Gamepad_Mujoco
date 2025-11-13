Gamepad Teleoperation Pipeline Documentation
============================================

Welcome to the Robotic Arm Teleoperation System documentation.

This system provides gamepad-based teleoperation for various robotic arms in MuJoCo simulation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   architecture
   usage

Overview
--------

The teleoperation system supports multiple robotic arms:

- **Panda & UR5**: Full 6DOF inverse kinematics control
- **SO100**: Hybrid joint-space + position-only IK control

Quick Start
-----------

.. code-block:: bash

   python gamepad_control.py panda

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`