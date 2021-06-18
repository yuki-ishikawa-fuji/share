#!/usr/bin/env python
# coding: utf-8

## Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import os



## Define functions
def FTCS(u_array, Time_step):
    """
    FTCS (Forward in Time and Central Difference in Space) scheme:
    u(n+1, j) = u(n, j) - CFL/2 (u(n, j+1) - u(n, j-1))
    
    Args
    ------
    u_array: initial square wave [numpy.ndarray]
    Time_step: time step of the calculation
    
    Returns
    ----------
    u_ftcs: numerically calculated wave by FTCS after the given time [numpy.ndarray]
    """ 
    u_ftcs = u_array.copy()
    for n in range(Time_step):
        u_old = u_ftcs.copy()
        u_ftcs[0] = u_old[0] - CFL / 2 * (u_old[1] - u_lower_boundary)
        u_ftcs[-1] = u_old[-1] - CFL / 2 * (u_upper_boundary - u_old[-1])
        for j in range(1, len(u_array)-1):
            u_ftcs[j] = u_old[j] - CFL / 2 * (u_old[j+1] - u_old[j-1])
    return u_ftcs


def LW(u_array, Time_step):
    """
    Lax-Wendroff scheme:
    u(n+1, j) = u(n, j) - CFL/2 (u(n, j+1) - u(n, j-1)) + CFL^2 / 2 (u(n, j+1) 2u(n, j) - u(n, j-1))
    
    Args
    ------
    u_array: initial square wave [numpy.ndarray]
    Time_step: time step of the calculation
    
    Returns
    ----------
    u_lw: numerically calculated wave by Lax-Wendroff scheme after the given time [numpy.ndarray]
    """ 
    u_lw= u_array.copy()
    for n in range(Time_step):
        u_old = u_lw.copy()
        u_lw[0] = u_old[0] - CFL / 2 * (u_old[1] - u_lower_boundary)                         + CFL**2 / 2 * (u_old[1] - 2 * u_old[0] + u_lower_boundary)
        u_lw[-1] = u_old[-1] - CFL / 2 * (u_upper_boundary - u_old[-1])                         + CFL**2 / 2 * (u_upper_boundary - 2 * u_old[-1] + u_old[-2])
        for j in range(1, len(u_array)-1):
            u_lw[j] = u_old[j] - CFL / 2 * (u_old[j+1] - u_old[j-1])                         + CFL**2 / 2 * (u_old[j+1] - 2 * u_old[j] + u_old[j-1])
    return u_lw


def CIP(u_array, Time_step):
    """
    CIP (Constrained Interpolation Profile) method:
    
    Args
    ------
    u_array: initial square wave [numpy.ndarray]
    Time_step: time step of the calculation
    
    Returns
    ----------
    u_cip: numerically calculated wave by CIP method after the given time [numpy.ndarray]
    """ 
    u_cip= u_array.copy()
    partial_u_cip = ((np.append(u_cip[1:], u_upper_boundary) + u_cip)/2 - (np.append(u_lower_boundary, u_cip[:-1]) + u_cip)/2)/ Delta_x
    for n in range(Time_step):
        u_old = u_cip.copy()
        partial_u_old = partial_u_cip.copy()
        u_cip[0] = 0
        partial_u_cip[0] = 0
        for j in range(1, len(u_array)):
            a = (partial_u_old[j] + partial_u_old[j-1]) / Delta_x**2 - 2.0 * (u_old[j] - u_old[j-1]) / Delta_x**3
            b = 3 * (u_old[j-1] - u_cip[j]) / Delta_x**2 + (2.0*partial_u_old[j] + partial_u_old[j-1]) / Delta_x
            c = partial_u_old[j]
            d = u_old[j]
            xi = - C * Delta_t  # C > 0
            u_cip[j:j+1] = a * xi**3 + b * xi**2 + c * xi + d
            partial_u_cip[j:j+1] = 3 * a * xi**2 + 2 * b * xi + c
    return u_cip


def mkplot(x_array, u_array, label, title, figname):
    """
    Visualize the exact and numerical answer of advection equation
    """
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)    
    ax.plot(x_array, exact_u_array, label="Answer")
    ax.plot(x_array, u_array, label=label)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("u", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_xlim(0, max(x_array))
    ax.set_ylim(-0.5,1.5)
    plt.title(title, fontsize=16)
    if not os.path.isdir("./Figures"):
        os.mkdir("./Figures")
    plt.savefig(f"./Figures/{figname}")



## Main program
if __name__ == "__main__":
    Num_stencil_x = 101
    x_array = np.arange(Num_stencil_x)
    u_array = np.where(((x_array >= 30) |(x_array < 10)), 0.0, 1.0) # initial square wave
    u_lower_boundary = 0.0
    u_upper_boundary = 0.0
    
    # Initial condition 1
    condition = "Condition 1"
    Time_step = 200
    C = 1
    Delta_t = 0.2
    
    # Initial condition 2
#     condition = "Condition 2"
#     Time_step = 40
#     C = 5
#     Delta_t = 0.2
    
    # Initial condition 3
#     condition = "Condition 3"
#     Time_step = 20
#     C = 10
#     Delta_t = 0.2
    
    Delta_x = max(x_array) / (Num_stencil_x-1)
    CFL = C * Delta_t / Delta_x
    total_movement = C * Delta_t * (Time_step+1)
    
    # Exact solution
    exact_u_array = np.where(((x_array >= 30 + total_movement) |(x_array < 10 + total_movement)), 0.0, 1.0)
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.plot(x_array, u_array, label="Initial condition", color="tab:red")
    ax.plot(x_array, exact_u_array, label="Answer", color="tab:blue")
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("u", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_xlim(0, max(x_array))
    ax.set_ylim(-0.5,1.5)
    plt.title("Initial Wave", fontsize=16)
    plt.savefig(f"./Figures/01_Exact.png")
    
    # FTCS
    u_ftcs = FTCS(u_array, Time_step)
    mkplot(x_array, u_ftcs, label="FTCS", title=f"FTCS ({condition})", figname=f"01_FTCS_{condition[-1]}.png")
    
    # Lax-Wendroff 
    u_lw = LW(u_array, Time_step)
    mkplot(x_array, u_lw, label="Lax-Wendroff", title=f"Lax-Wendroff  ({condition})", figname=f"01_Lax-Wendroff_{condition[-1]}.png")
    
    # CIP
    u_cip = CIP(u_array, Time_step)
    mkplot(x_array, u_cip, label="CIP", title=f"CIP ({condition})", figname=f"01_CIP_{condition[-1]}.png")