import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

class Body:
    def __init__(self, name, mass, position, velocity, color, trail_length=200):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype='float64')  # [x, y]
        self.velocity = np.array(velocity, dtype='float64')  # [vx, vy]
        self.color = color
        self.trajectory = []
        self.trail_length = trail_length  # Max length of the trail

    def update(self, force, dt):
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        self.trajectory.append(self.position.copy())

        # Limit the length of the trajectory
        if len(self.trajectory) > self.trail_length:
            self.trajectory.pop(0)

def gravitational_force(body1, body2):
    G = 6.67430e-11  # gravitational constant
    r_vector = body2.position - body1.position
    distance = np.linalg.norm(r_vector)
    if distance == 0:
        return np.array([0.0, 0.0])
    force_magnitude = G * body1.mass * body2.mass / distance**2
    force_direction = r_vector / distance
    return force_direction * force_magnitude

# Simulation parameters
AU = 1.496e+11  # Astronomical Unit in meters
DAY = 24 * 3600  # One day in seconds
dt = 1.0 * DAY  # Increase time step for faster simulation

# Initialize bodies
sun     = Body("Sun", 1.989e30, [0, 0], [0, 0], 'yellow')
earth   = Body("Earth", 5.972e24, [AU, 0], [0, 29_780], 'blue')
moon    = Body("Moon", 7.34767309e22, [AU + 384_400_000, 0], [0, 29_780 + 1_022], 'lightgray')
mercury = Body("Mercury", 3.3011e23, [0.387 * AU, 0], [0, 47_360], 'gray')
venus   = Body("Venus", 4.8675e24, [0.723 * AU, 0], [0, 35_020], 'orange')
mars    = Body("Mars", 6.4171e23, [1.524 * AU, 0], [0, 24_077], 'red')
jupiter = Body("Jupiter", 1.898e27, [5.204 * AU, 0], [0, 13_070], 'gold', trail_length=200)

bodies  = [sun, mercury, venus, earth, mars, jupiter, moon]

# Set up figure
fig, ax = plt.subplots(figsize=(6.4, 8), dpi=150)
ax.set_facecolor("black")
ax.set_aspect('equal')
fig.patch.set_facecolor('black')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')

# Adjust view to fit Jupiter
ax.set_xlim(-6 * AU, 6 * AU)
ax.set_ylim(-6 * AU, 6 * AU)

# Grid, titles, and labels
ax.grid(False)
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")

# Set white axes
ax.tick_params(colors='white')

# Define marker sizes for each body
sizes = {
    "Sun": 15,
    "Mercury": 3,
    "Venus": 5,
    "Earth": 6,
    "Moon": 2,
    "Mars": 4,
    "Jupiter": 8
}

# Create plot objects
scatters = [ax.plot([], [], 'o', color=body.color, markersize=sizes.get(body.name, 6), label=body.name)[0] for body in bodies]
trajectories = [ax.plot([], [], '-', color=body.color, alpha=0.3)[0] for body in bodies]

# Legend
legend = ax.legend(loc="upper right", fontsize=8)
frame = legend.get_frame()
frame.set_facecolor('black')
frame.set_edgecolor('white')
for text in legend.get_texts():
    text.set_color('white')

# Animation update function
def update(frame):
    forces = [np.zeros(2) for _ in bodies]

    # Calculate forces
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i != j:
                forces[i] += gravitational_force(body1, body2)

    # Update bodies
    for i, body in enumerate(bodies):
        body.update(forces[i], dt)

    # Update visuals
    for i, body in enumerate(bodies):
        scatters[i].set_data([body.position[0]], [body.position[1]])
        traj = np.array(body.trajectory)
        if len(traj) > 1:
            trajectories[i].set_data(traj[:, 0], traj[:, 1])
            alpha_values = np.linspace(0.1, 0.5, len(traj))  # Decrease alpha for earlier points
            trajectories[i].set_alpha(0.5)  # Set constant max alpha
            trajectories[i].set_color(body.color)
            trajectories[i].set_alpha(0.3)  # Apply to the whole line

    return scatters + trajectories

ani = FuncAnimation(fig, update, frames=800, interval=20, blit=True)

writer = FFMpegWriter(fps=30, metadata=dict(artist='Dante Stellare'), bitrate=1800)
ani.save("solar_system_simulation.mp4", writer=writer)
