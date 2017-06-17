#!/usr/bin/python2.7
import numpy as np
from scipy.fftpack import fft as sc_fft
from scipy.fftpack import ifft as sc_ifft
from cubic_hermite_spline import MonotoneCubicInterpolation as cubic_interp


def green_inv_fourier(Giwn, beta, M, tau_mesh):
    wn_mesh = len(Giwn)
    assert(len(M) >= 3)
    assert(tau_mesh/2. >= wn_mesh)

    iwn_edge = 1j*(2*wn_mesh - 1.)*np.pi/beta

    dist = lambda g_fin : np.abs(g_fin - (M[0]/iwn_edge\
                + M[1]/iwn_edge**2 + M[2]/iwn_edge**3))

    if dist(Giwn[-1]) >= 1e-7:
        raise ValueError(\
              "warning! enlarge your size of the Matsubara\
               frequency region!(fourier transform can give\
               a result with a large error)")

    iwn = np.array([1j*(2.*n + 1.)*np.pi/beta for n in\
                 xrange(wn_mesh)], dtype = 'complex128')

    giwn = np.array(Giwn.copy())

    giwn = Giwn - (M[0]/iwn + M[1]/iwn**2 + M[2]/iwn**3)

    gtau = np.zeros([wn_mesh], dtype = 'complex128'); gtau_temp = sc_fft(giwn[:-1]); gtau[:-1] = gtau_temp

    tau = np.linspace(0, beta, wn_mesh)

    gtau = 2./beta*np.exp(-1j*np.pi*tau/beta)*gtau\
           - 0.5*M[0] + (tau/2. - beta/4.)*M[1] + (tau*beta/4. - tau**2/4.)*M[2]

    gtau[-1] = 2./beta*np.sum(-giwn) - 0.5*M[0] + (beta/4.)*M[1]

    # monotone cubic interpolation.
    interp = cubic_interp(tau, gtau.real)

    tau = np.linspace(0, beta, tau_mesh)
    gtau = np.zeros([tau_mesh])

    for i, tau_i in enumerate(tau):
        gtau[i] = interp(tau_i)

    return gtau


def green_fourier(Gtau, beta, wn_mesh):
    tau_mesh = len(Gtau)
    assert(tau_mesh/2 >= wn_mesh)
    dtau = beta/(tau_mesh - 1.)

    assert(tau_mesh%2 == 1)

    tau = np.linspace(0, beta, tau_mesh)
    gtemp = Gtau*np.exp(1j*np.pi/beta*tau)
   
    for i in xrange(1, tau_mesh-1, 2):
        gtemp[i] *= 4.
    for i in xrange(2, tau_mesh-2, 2):
        gtemp[i] *= 2.
    gtemp *= 1./3.*dtau
    
    giwn_temp = np.zeros([tau_mesh], dtype = 'complex128')

    giwn_temp[:-1] = tau_mesh*sc_ifft(gtemp[:-1])

    giwn = np.zeros([wn_mesh], dtype = 'complex128'); giwn[:wn_mesh] = giwn_temp[:wn_mesh]

    giwn += gtemp[-1]

    return giwn




if __name__ == "__main__":
    wn_mesh = int(1e3)
    tau_mesh = int(1e4+1)
    beta = 100.

    iwn = np.array([1j*(2.*n + 1.)*np.pi/beta for n in\
                 xrange(wn_mesh)], dtype = 'complex128')

    giwn = 1./(iwn - 1.)

    gtau = green_inv_fourier(giwn, beta, [1, 1, 1], tau_mesh)

    giwn_re = green_fourier(gtau, beta, wn_mesh)
