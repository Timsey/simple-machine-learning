# CartPole physics

The CartPole system is an interesting exercise in classical mechanics. Here I provide a derivation of the system's equations of motion using Lagrangian mechanics.

# System definition

The CartPole system consists of a frictionlessly moving cart, connected to an inverted physical pendulum. Standing fully upright, the pendulum is in an unstable state, meaning that any perturbation will cause it to start falling.

By initialising the system already in motion, with the pendulum on some small offset from the vertical, we can train an agent to balance the pole. The agent is allowed to move left or right with some (usually predefined) force F, and its goal is to prevent the pole from falling for a given number of timesteps in the simulation.

In order to derive the equations of motion, we first define the variables in our system:

<p align="center">
  <img src="images/CartPole.png" width="350" title="Diagram of CartPole system.">
</p>

- x: the x position of the pole's center of mass.
- y: the y position of the pole's center of mass.
- M: the mass of the cart.
- m: the mass of the pole.
- L: half the length of the pole.
- F: the force on the cart (this is the force the learning agent will exert on the cart in a timestep).
- $\theta$: the angle between the y axis and the pole.
- $\dot{\theta}$: the angular velocity of the pole.
- $\dot{x}$: the velocity of the CartPole system in the x direction.
- $v_x$: the velocity of the pole in the x direction as a result of rotation.
- $v_y$: the velocity of the pole in the y direction as a result of rotation.

Our basic reinforcement learning agent will only have access to x, $\theta$, $\dot{\x}$ and $\dot{\theta}$ as the state variable.