# CartPole physics

The CartPole system is an interesting exercise in classical mechanics. Here we provide a derivation of the system's equations of motion using Lagrangian mechanics.

## System definition

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
- $v_x$: the velocity of the pole in the x direction as a result of rotation and the movement of the cart.
- $v_y$: the velocity of the pole in the y direction as a result of rotation.

Our basic reinforcement learning agent will only have access to x, $\theta$, $\dot{\x}$ and $\dot{\theta}$ as the state variable.

## Derivation 

Given the Lagrangian of this system, we can find the equations of motion by solving the Euler-Lagrange equations. The Lagrangian (L) is given by the sum of the kinetic (T) and potential energy (V) of the system.

$$ L = T + V $$

The potential energy is simply the gravitational potential energy of the falling pole. For any angle $/theta$ the height (h) of the center of mass of the pole is given by $L \cos{\theta}$. The gravitational potential energy $V = m g h = m g \cost{\theta}$, where g is the  gravitational acceleration at the Earth's surface (approximately $9.81 m / s^2$).

The kinetic energy of this system consists of two parts:
1) the kinetic energy of the cart, due to its velocity $\dot{x}$.
2) the kinetic energy of the pole, due to both the cart moving, and the rotation caused by falling.

A body of mass m moving with velocity v has a translational kinetic energy equal to $\frac{1}{2} m v^2$. In the case of our cart, this translates to a kinetic energy of $T_cart = \frac{1}{2} M \dot{x}^2$.

Our pole has two kinetic energy components: translational and rotational. The translational kinetic energy is given by the same formula as before. However, v now has two components: we introduce the following diagram for clarity:

<p align="center">
  <img src="images/PoleVs.png" width="350" title="Translational velocities of the pole.">
</p>

Note that the diagram doesn't yet include the translational velocity $\dot{x}$. The total translational velocity $v_pole$ is given by the Pythagorean theorem as $v_pole^2 = v_pole_x^2 + v_pole_y^2$. 

From the diagram we read that $v_pole_y = v_y = L \dot{\theta} \sin{\theta}$, and $v_pole_x = \dot{x} + v_x = \dot{x} + L \dot{\theta} \cos{\theta}$. Thus we find that:

$$v^2 = L^2 \dot{\theta}^2 + \dot{x}^2 + 2 L \cos{\theta} \dot{x} \dot{\theta}$$