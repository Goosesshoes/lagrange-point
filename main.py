import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numba import njit

solar_mass = 1.9891e30
earth_mass = 5.9722e24
earth_velocity = np.array([0,30e3])
solar_cords = np.array([0,0])
earth_cords = np.array([149.6e9,0])
G = 6.6743e-11
stepsize = 1000


def earthupdate():
    global earth_cords,earth_velocity
    for i in range(100):
        distance = np.sqrt((solar_cords[0]-earth_cords[0])**2+(solar_cords[1]-earth_cords[1])**2)

        force = (G*solar_mass*earth_mass)/(distance**2)
        A_earth = force/earth_mass 

        A_dir = (solar_cords-earth_cords)/distance
        Vchange = A_earth*A_dir
        earth_velocity+=Vchange*stepsize

        earth_cords+= earth_velocity*stepsize
    return earth_cords

fig, ax = plt.subplots()
scat = ax.scatter([], [])

ax.set_xlim(-2e11, 2e11)
ax.set_ylim(-2e11, 2e11)

def update(frame):
    pos = earthupdate()

    scat.set_offsets(np.c_[[pos[0],0], [pos[1],0]])
    return scat,


ani = animation.FuncAnimation(fig, update, interval=0.02)
plt.show()