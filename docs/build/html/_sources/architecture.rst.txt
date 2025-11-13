System Architecture
===================

Design Overview
---------------

The system uses a modular architecture with clear separation between configuration, control logic, and simulation.

Complete System Architecture
----------------------------

.. graphviz::

   digraph architecture {
       rankdir=TB;
       node [shape=rectangle];
       
       // Configuration and Initialization
       Config [label="Configuration Manager"];
       Config -> GamepadControlLauncher [label="robot config"];
       
       // User Input and Selection
       Gamepad -> GamepadControlLauncher;
       GamepadControlLauncher -> GenericTeleoperation [label="Panda/UR5"];
       GamepadControlLauncher -> SO100Teleoperation [label="SO100"];
       
       // Core Control Systems
       GenericTeleoperation -> GenericVelocityIKSolver;
       GenericTeleoperation -> MovementHelper;
       
       SO100Teleoperation -> SO100IKSolver;
       SO100Teleoperation -> SO100JointController;
       SO100Teleoperation -> MovementHelper;
       
       // Simulation Layer
       GenericVelocityIKSolver -> Simulation;
       SO100IKSolver -> Simulation;
       MovementHelper -> Simulation;
       SO100JointController -> Simulation;
       
       // Configuration Dependencies (dashed lines)
       Config -> GenericTeleoperation [style=dashed, label="params"];
       Config -> SO100Teleoperation [style=dashed, label="params"];
       Config -> GenericVelocityIKSolver [style=dashed, label="joint mapping"];
       Config -> MovementHelper [style=dashed, label="scales"];
   }