#!/usr/bin/env python
# coding: utf-8

## Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time



## Define functions
def CenteredDiff(temperature_array, Time_step):
    """
    (Second order) centered difference scheme [Explicit scheme]:
    T(n+1, j) = T(n, j) + κ * Δt / Δx^2 (T(n, j+1) -2T(n, j) +T(n, j-1))
    
    Args
    ------
    temperature_array: initial condition of temperature [numpy.ndarray]
    Time_step: time step of the calculation
    
    Returns
    ----------
    temperature_explicit: numerically calculated tempearature by Centered diffrence after the given time step [numpy.ndarray]
    """
    
    temperature_explicit = temperature_array.copy()
    for n in range(Time_step):
        temperature_old = temperature_explicit.copy()
        temperature_explicit[0] += kappa * Delta_t / Delta_x**2 *             (temperature_explicit[1] - 2*temperature_old[0] + temperature_lower_boundary)
        temperature_explicit[-1] += kappa * Delta_t / Delta_x**2 *             (temperature_upper_boundary - 2*temperature_old[-1] + temperature_old[-2])
        for j in range(1, len(temperature_array)-1):
            temperature_explicit[j] += kappa * Delta_t / Delta_x**2 *                 (temperature_old[j+1] - 2*temperature_old[j] + temperature_old[j-1])
    return temperature_explicit


def CNmatrix(temperature_array_n):
    """
    Create the tridiagonal matrix (A) and the column vector (b)of Crank-Nicolson method [Ax = b]
    
    Args
    ------
    temperature_array_n: an array of temperature at n-th step [numpy.ndarray]
    
    Returns
    ----------
    a_matrix: the tridiagonal matrix of Crank-Nicolson method
    b_array: the column vector of Crank-Nicolson method
    """
    a_matrix = np.identity(len(temperature_array_n)) * 2 *(1/d+1)                 - np.eye(len(temperature_array_n), k=1) - np.eye(len(temperature_array_n), k=-1)
    temp_temperature_array = np.append(np.append(
                        temperature_lower_boundary, 
                        temperature_array_n), temperature_upper_boundary)
    b_array = 2 * (1/d - 1) * temperature_array_n + temp_temperature_array[2:] + temp_temperature_array[:-2]
    b_array[0] += temperature_lower_boundary
    b_array[-1] += temperature_upper_boundary
    return a_matrix, b_array


def LU(temperature_array, Time_step):
    """
    LU factorization [Implicit scheme ; Direct method]:
    
    Args
    ------
    temperature_array: initial condition of temperature [numpy.ndarray]
    Time_step: time step of the calculation
    
    Returns
    ----------
    temperature_lu: numerically calculated tempearature by LU factorization after the given time step [numpy.ndarray]
    elapsed_time: calculation time [s]
    """
    t0 = time.time()
    m = len(temperature_array)
    temperature_lu = temperature_array.copy()
    for n in range(Time_step):
        a_matrix, b_array = CNmatrix(temperature_lu)
        U_matrix = a_matrix.copy()
        L_matrix = np.identity(m)
        for k in range(m-1):
            for j in range(k+1, m):
                L_matrix[j][k] = U_matrix[j][k] / U_matrix[k][k]
                U_matrix[j][k:m] -= L_matrix[j][k] * U_matrix[k][k:m]
        x = np.zeros(m)
        y = np.zeros(m)
        # forward substitution (Ly=b)
        for k in range(m):
            if k == 0: 
                y[k] = b_array[k] / L_matrix[k][k]
            else:
                y[k] = (b_array[k] - np.sum(np.multiply(L_matrix[k][0:k], y[0:k]))) / L_matrix[k][k]
        # backward substitution (Ux = y)
        for k in range(m-1, -1, -1):
            if k == m-1: 
                x[k] = y[k] / U_matrix[k][k]
            else:
                x[k] = (y[k] - np.sum(np.multiply(U_matrix[k][k+1:m], x[k+1:m]))) / U_matrix[k][k]
        temperature_lu = x.copy()
    elapsed_time = time.time() - t0
    return temperature_lu, elapsed_time


def TDMA(temperature_array, Time_step):
    """
    Triagonal matrix algorithm (Thomas algorithm) [Implicit scheme ; Direct method]:
    
    Args
    ------
    temperature_array: initial condition of temperature [numpy.ndarray]
    Time_step: time step of the calculation
    
    Returns
    ----------
    temperature_tdma: numerically calculated tempearature by Thomas algorithm after the given time step [numpy.ndarray]
    elapsed_time: calculation time [s]
    """
    t0 = time.time()
    m = len(temperature_array)
    temperature_tdma = temperature_array.copy()
    for n in range(Time_step):
        temperature_old = temperature_tdma.copy()
        a_array = np.ones(m) * (-1) # an array containing lower diagonal (a[0] is not used)
        b_array = np.ones(m) * 2 * (1/d + 1) # an array containing main diagonal of A
        c_array = np.ones(m) * (-1) # an array containing lower diagonal (c[-1] is not used)
        temp_temperature_array = np.append(np.append(temperature_lower_boundary, temperature_old), temperature_upper_boundary)
        d_array = 2 * (1/d - 1) * temperature_old + temp_temperature_array[2:] + temp_temperature_array[:-2] # right hand side of the system
        d_array[0] += temperature_lower_boundary
        d_array[-1] += temperature_upper_boundary
        #elimination
        for k in range(1,m):
            q = a_array[k] / b_array[k-1]
            b_array[k] = b_array[k] - c_array[k-1]*q
            d_array[k] = d_array[k] - d_array[k-1]*q
        # backsubstitution:
        q = d_array[m-1]/b_array[m-1]
        temperature_tdma[m-1] = q
        for k in range(m-2,-1,-1):
            q = (d_array[k]-c_array[k]*q)/b_array[k]
            temperature_tdma[k] = q
    elapsed_time = time.time() - t0
    return temperature_tdma, elapsed_time


def Jacobi(temperature_array, Time_step, tol=1e-8):
    """
    Jacobi method [Implicit scheme ; Iterative method]:
    
    Args
    ------
    temperature_array: initial condition of temperature [numpy.ndarray]
    Time_step: time step of the calculation
    tol: error tolerance
    
    Returns
    ----------
    temperature_jacobi: numerically calculated tempearature by Jacobi after the given time step [numpy.ndarray]
    elapsed_time: calculation time [s]
    """
    t0 = time.time()
    m = len(temperature_array)
    temperature_jacobi = temperature_array.copy()
    for n in range(Time_step):
        a_matrix, b_array = CNmatrix(temperature_jacobi)
        x = b_array.copy()
        x_old = b_array.copy()
        diag_matrix= np.diag(a_matrix)
        l_u_matrix = a_matrix - np.diagflat(diag_matrix) 
        count = 0
        while True:
            count += 1
            x = (b_array - np.dot(l_u_matrix, x_old))/diag_matrix
            residual = np.linalg.norm(x - x_old) / np.linalg.norm(x)
            x_old = x
            if residual <= tol:
                break
            elif count >= 10000:
                print(residual)
                sys.exit()
        temperature_jacobi = x
    elapsed_time = time.time() - t0
    return temperature_jacobi, elapsed_time


def mkplot(x_array, temperature_list, label_list, title, figname):
    """
    Visualize the exact and numerical answer of advection equation
    """
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)    
    ax.plot(x_array, exact_temperature_array, label="Answer", color="black")
    for temperature_array, label in zip(temperature_list, label_list):
        ax.plot(x_array, temperature_array, label=label)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("Temperature [K]", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(x_array))
    ax.set_ylim(100, 200)
    plt.title(title, fontsize=16)
    if not os.path.isdir("./Figures"):
        os.mkdir("./Figures")
    plt.savefig(f"./Figures/{figname}")



## Main program
if __name__ == "__main__":
    Num_stencil_x = 101
    x_array = np.float64(np.arange(Num_stencil_x))
    temperature_array = x_array + 100
    temperature_lower_boundary = 150
    temperature_upper_boundary = 150
    Delta_x = max(x_array) / (Num_stencil_x-1)
    
    # Initial condition 1
    condition = "Condition 1"
    Delta_t = 0.2
    kappa = 0.5
    d = kappa * Delta_t / Delta_x**2
    
    # Initial condition 2
#     condition = "Condition 2"
#     Delta_t = 0.2
#     kappa = 3
#     d = kappa * Delta_t / Delta_x**2

    # Exact solution
    exact_temperature_array = (temperature_upper_boundary - temperature_lower_boundary) / (x_array[-1] - x_array[0]) * x_array + temperature_lower_boundary
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)    
    ax.plot(x_array, exact_temperature_array, label="Answer", color="black")
    ax.plot(x_array, temperature_array, label="Initial condition", color="red")
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("Temperature [K]", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(x_array))
    ax.set_ylim(100,200)
    plt.title("Initial temperature distribution", fontsize=16)
    plt.savefig(f"./Figures/02_Exact.png")
    
    # Centered Difference
    temperature_explicit1 = CenteredDiff(temperature_array, Time_step=100)
    temperature_explicit2 = CenteredDiff(temperature_array, Time_step=1000)
    temperature_explicit3 = CenteredDiff(temperature_array, Time_step=10000)
    temperature_explicit_list = [temperature_explicit1, temperature_explicit2, temperature_explicit3]
    label_explicit_list = ["Explicit (100 steps)", "Explicit (1000 steps)", "Explicit (10000 steps)"]
    mkplot(x_array, temperature_explicit_list, label_explicit_list, title=f"Centered difference ({condition})", figname=f"02_CD_{condition[-1]}.png")
    
    # LU Factorization
    temperature_lu1, time_lu1 = LU(temperature_array, Time_step=100)
    temperature_lu2, time_lu2 = LU(temperature_array, Time_step=1000)
    temperature_lu3, time_lu3 = LU(temperature_array, Time_step=10000)
    temperature_lu_list = [temperature_lu1, temperature_lu2, temperature_lu3]
    label_lu_list = ["Implicit (LU: 100 steps)", "Implicit (LU: 1000 steps)", "Implicit (LU: 10000 steps)"]
    mkplot(x_array, temperature_lu_list, label_lu_list, title=f"LU factorization ({condition})", figname=f"02_LU_{condition[-1]}.png")
    
    # TDMA
    temperature_tdma1, time_tdma1 = TDMA(temperature_array, Time_step=100)
    temperature_tdma2, time_tdma2 = TDMA(temperature_array, Time_step=1000)
    temperature_tdma3, time_tdma3 = TDMA(temperature_array, Time_step=10000)
    temperature_tdma_list = [temperature_tdma1, temperature_tdma2, temperature_tdma3]
    label_tdma_list = ["Implicit (TDMA: 100 steps)", "Implicit (TDMA: 1000 steps)", "Implicit (TDMA: 10000 steps)"]
    mkplot(x_array, temperature_tdma_list, label_tdma_list, title=f"TDMA algorithm ({condition})", figname=f"02_TDMA_{condition[-1]}.png")
    
    # Jacobi method
    temperature_jacobi1, time_jacobi1 = Jacobi(temperature_array, Time_step=100)
    temperature_jacobi2, time_jacobi2 = Jacobi(temperature_array, Time_step=1000)
    temperature_jacobi3, time_jacobi3 = Jacobi(temperature_array, Time_step=10000)
    temperature_jacobi_list = [temperature_jacobi1, temperature_jacobi2, temperature_jacobi3]
    label_jacobi_list = ["Implicit (Jacobi: 100 steps)", "Implicit (Jacobi: 1000 steps)", "Implicit (Jacobi: 10000 steps)"]
    mkplot(x_array, temperature_jacobi_list, label_jacobi_list, title=f"Jacobi method ({condition})", figname=f"02_Jacobi_{condition[-1]}.png")


    # Check the calculation time for each method and step
    print(f"LU (100 steps): {time_lu1}　[s]")
    print(f"LU (1000 steps): {time_lu2}　[s]")
    print(f"LU (10000 steps): {time_lu3}　[s]")
    print(f"TDMA (100 steps): {time_tdma1}　[s]")
    print(f"TDMA (1000 steps): {time_tdma2}　[s]")
    print(f"TDMA (10000 steps): {time_tdma3}　[s]")
    print(f"Jacobi (100 steps): {time_jacobi1}　[s]")
    print(f"Jacobi (1000 steps): {time_jacobi2}　[s]")
    print(f"Jacobi (10000 steps): {time_jacobi3}　[s]")


    # Visualize the calculation time
    step = [100, 1000, 10000]
    time_lu_list = [time_lu1, time_lu2, time_lu3]
    time_tdma_list = [time_tdma1, time_tdma2, time_tdma3]
    time_jacobi_list = [time_jacobi1, time_jacobi2, time_jacobi3]
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111) 
    ax.plot(step, time_lu_list, label="LU", color="red")
    ax.plot(step, time_tdma_list, label="TDMA", color="blue")
    ax.plot(step, time_jacobi_list, label="Jacobi", color="green")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Calculation step", fontsize=14)
    ax.set_ylabel("Time [s]", fontsize=14)
    ax.legend(loc="upper left")
    plt.title(f"{condition}", fontsize=16)
    plt.savefig(f"./Figures/02_CalculationTime_{condition[-1]}.png")
