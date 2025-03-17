# learn-reinforcement-learning

Playground to learn RL concepts.

## Good Gym Envs to Learn Discrete Algos
### Blackjack-v0
#### Acion
Discrete(2)
0=stand, 1=hit

#### State
Tuple(Discrete(32), Discrete(11), Discrete(2))
Player's hand, dealer's face up card, player has usable ace or not (0=yes, 1=no)

### FrozenLake-v0
#### Action
Discrete(4)
Action will be an integer between $[0, 4)$ with the following meanings:
  * LEFT = 0
  * DOWN = 1
  * RIGHT = 2
  * UP = 3
  
#### State
Discrete(16)
For a 4x4 grid, the state will be a single number between $[0, 16)$. This is the cell that the agent is in. As in all gridworld environments, the cells are numbered horizontally like so:
$$
00\;01\;02\;03 \\
04\;05\;06\;07 \\
08\;09\;10\;11 \\
12\;13\;14\;15 \\
$$

### CliffWalking-v0
#### Action
Discrete(4)
Action will be an integer between $[0, 4)$ with the following meanings:
  * UP = 0
  * RIGHT = 1
  * DOWN = 2
  * LEFT = 3
  
#### State
Discrete(48)
State will be a single number between $[0, 48)$. This the cell number that the agent is in. As in all gridworld environments, the cells are numbered horizontally like so:
$$
00\;01\;02\;03\;04\;05\;06\;07\;08\;09\;10\;11 \\
12\;13\;14\;15\;16\;17\;18\;19\;20\;21\;22\;23 \\
24\;25\;26\;27\;28\;29\;30\;31\;32\;33\;34\;35 \\
36\;37\;38\;39\;40\;41\;42\;43\;44\;45\;46\;47 \\
$$

### NChain-v0
#### Action
Discrete(2)
#### State
Discrete(5)

## Gym Envs for DQN
### CartPole-v0
The objective of the game is to keep the pole upright as long as possible. For every timestep the pole is upright, the agent gets a reward of +1, including the terminal state. The game ends when after 200 timesteps or when the pole tips $12^{\circ}$ on either side.

#### Action
Discrete(2)
Action will be an integer between $[0, 2)$ with the following meaning:
  * LEFT = 0
  * RIGHT = 1

#### State
Box(4,)
This just means that the state will be a vector with 4 elements with the following meanings:
  * $[0] = x$
  Cart position on the x-axis. The origin of the x-axis is at the center of the railing. The limits are $[-4.8, +4.8]$, i.e, the edges of the frame are 4.8 units away from the origin.

  * $[1] = \dot x$
  Cart velocity along the x-axis. This can range from $[-\infty, +\infty]$

  * $[2] = \theta$
  Pole angle. This ranges from $(-24^{\circ}$, $24^{\circ})$.

  * $[3] = \dot \theta$
  The documentation says pole velocity at tip. Not sure whether this is the angular velocity or the linear velocity. But it can range from $[-\infty, +\infty]$

## LunarLander-v2
The objective of this game is to land the Lander on the landing pad - marked by two flag poles, without crashing and using minimum fuel. Agent can fire the vertical thruster or the side thrusters to align the Lander correctly, but every time it does that it loses points, $-0.3$ for firing the main vertical thruster and $-0.03$ for the side thruster. The coordinates of the landing pad are $(0, 0)$. Every episode of the game has a slightly different terrain and landing pad position. But regardless of where the landing pad is in the frame, it is always the origin.

### Action
Discrete(4)
The action will be an integer between $[0, 4)$ with the following meaning:
  * NOOP = 0
  * FIRE LEFT = 1
  * FIRE MAIN = 2
  * FIRE RIGHT = 3

### State
Box(8,)
The state is a vector of 8 elements with the following meanings:
  * $[0] = x$
  The x coordinate of the lander.

  * $[1] = y$
  The y coordinate of the lander.

  * $[2] = \dot x$
  Velocity of the lander w.r.t x-axis.

  * $[3] = \dot y$
  Velocity of the lander w.r.t y-axis.

  * $[4] = \theta$
  Angle of the lander. I think this is w.r.t the y-axis.

  * $[5] = \dot \theta$
  Angular velocity of the Lander.

  * $[6] = \text{Left leg contact}$
  This is $1$ if the left leg is in contact with the ground, $0$ otherwise.

  * $[7] = \text{Right leg contact}$
  This $1$ if the right leg is in contact with the ground, $0$ otherwise.