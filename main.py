import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numba import njit

solar_mass = 1e30
earth_mass = 5e27
earth_velocity = np.array([0,2.11e4])
solar_cords = np.array([0,0])
earth_cords = np.array([149.6e9,0])
G = 6.6743e-11
stepsize = 100
angle = 1.04
astroid_cords = np.array([
            [np.cos(angle)*149.6e9,np.sin(angle)*149.6e9],
            [-149.6e9,0],
            [0,-149.6e9]
        ])


astroid_velocity = np.array([
            [-np.sin(angle)*2.11e4,np.cos(angle)*2.11e4],
            [0,-2.11e4],
            [2.11e4,0]
        ],dtype=np.float64)
mass_astroid = 10

def earthupdate():
    global earth_cords, earth_velocity,astroid_cords,astroid_velocity
    earth_cords,earth_velocity,astroid_cords,astroid_velocity=mathloop(earth_cords,earth_velocity,solar_cords,solar_mass,earth_mass,G,stepsize,astroid_cords,astroid_velocity,mass_astroid)
    return earth_cords,astroid_cords
@njit
def mathloop(earth_cords,earth_velocity,solar_cords,solar_mass,earth_mass,G,stepsize,astroid_cords,astroid_velocity,mass_astroid):
    for i in range(100):
        distance = np.sqrt((solar_cords[0]-earth_cords[0])**2+(solar_cords[1]-earth_cords[1])**2)

        force = (G*solar_mass*earth_mass)/(distance**2)
        A_earth = force/earth_mass 

        A_dir = (solar_cords-earth_cords)/distance
        Vchange = A_earth*A_dir
        earth_velocity+=Vchange*stepsize

        earth_cords+= earth_velocity*stepsize



        asr_distance_sun = np.sqrt((solar_cords[0]-astroid_cords[:,0])**2+(solar_cords[1]-astroid_cords[:,1])**2)
        asr_distance_earth = np.sqrt((earth_cords[0]-astroid_cords[:,0])**2+(earth_cords[1]-astroid_cords[:,1])**2)
        asr_force_sun = (G*solar_mass*mass_astroid)/(asr_distance_sun**2)
        asr_force_earth = (G*earth_mass*mass_astroid)/(asr_distance_earth**2)
        asr_A_earth = asr_force_earth/mass_astroid
        asr_A_sun = asr_force_sun/mass_astroid
        asr_A_dir_sun = (solar_cords-astroid_cords)/asr_distance_sun[:,np.newaxis]
        asr_A_dir_earth = (earth_cords-astroid_cords)/asr_distance_earth[:,np.newaxis]
        asr_Vchange_earth = asr_A_earth[:, np.newaxis] *asr_A_dir_earth
        asr_Vchange_sun = asr_A_sun[:, np.newaxis] *asr_A_dir_sun
        asr_Vchange = asr_Vchange_earth+asr_Vchange_sun
        astroid_velocity+=asr_Vchange*stepsize
        astroid_cords+= astroid_velocity*stepsize
    return earth_cords,earth_velocity,astroid_cords,astroid_velocity








fig, ax = plt.subplots()
sun = ax.scatter([0],[0], s=500, color='yellow', edgecolors='orange')
earth_scat = ax.scatter([], [], s=50, color='blue')
ast_scat = ax.scatter([], [], s=20, color='red')

ax.set_xlim(-2e11, 2e11)
ax.set_ylim(-2e11, 2e11)

def update(frame):
    pos_earth,pos_astroids = earthupdate()


    pos = np.vstack([pos_earth.reshape(1,2), pos_astroids])

    earth_scat.set_offsets(pos_earth)
    ast_scat.set_offsets(pos_astroids)
    return earth_scat, ast_scat
 

ani = animation.FuncAnimation(fig, update, interval=50)
plt.show()
