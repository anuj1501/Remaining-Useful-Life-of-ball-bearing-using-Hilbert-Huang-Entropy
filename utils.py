import numpy as np
from scipy.signal import argrelmax, argrelmin
from scipy import interpolate, angle


def inst_freq(x, t=None):

    if x.ndim != 1:
        if 1 not in x.shape:
            raise TypeError("Input should be a one dimensional array.")
        else:
            x = x.ravel()
    if t is not None:
        if t.ndim != 1:
            if 1 not in t.shape:
                raise TypeError("Time instants should be a one dimensional "
                                "array.")
            else:
                t = t.ravel()
    else:
        t = np.arange(2, len(x))

    fnorm = 0.5 * (angle(-x[t] * np.conj(x[t - 2])) + np.pi) / (2 * np.pi)
    return fnorm, t


def boundary_conditions(signal, time_samples, z=None, nbsym=2):

    tmax = argrelmax(signal)[0]
    maxima = signal[tmax]
    tmin = argrelmin(signal)[0]
    minima = signal[tmin]

    if tmin.shape[0] + tmax.shape[0] < 3:
        raise ValueError("Not enough extrema.")

    loffset_max = time_samples[tmax[:nbsym]] - time_samples[0]
    roffset_max = time_samples[-1] - time_samples[tmax[-nbsym:]]
    new_tmax = np.r_[time_samples[0] - loffset_max[::-1],
                     time_samples[tmax], roffset_max[::-1] + time_samples[-1]]
    new_vmax = np.r_[maxima[:nbsym][::-1], maxima, maxima[-nbsym:][::-1]]

    loffset_min = time_samples[tmin[:nbsym]] - time_samples[0]
    roffset_min = time_samples[-1] - time_samples[tmin[-nbsym:]]

    new_tmin = np.r_[time_samples[0] - loffset_min[::-1],
                     time_samples[tmin], roffset_min[::-1] + time_samples[-1]]
    new_vmin = np.r_[minima[:nbsym][::-1], minima, minima[-nbsym:][::-1]]
    return new_tmin, new_tmax, new_vmin, new_vmax


def get_envelops(x, t=None):

    if t is None:
        t = np.arange(x.shape[0])
    maxima = argrelmax(x)[0]
    minima = argrelmin(x)[0]

    # consider the start and end to be extrema

    ext_maxima = np.zeros((maxima.shape[0] + 2,), dtype=int)
    ext_maxima[1:-1] = maxima
    ext_maxima[0] = 0
    ext_maxima[-1] = t.shape[0] - 1

    ext_minima = np.zeros((minima.shape[0] + 2,), dtype=int)
    ext_minima[1:-1] = minima
    ext_minima[0] = 0
    ext_minima[-1] = t.shape[0] - 1

    tck = interpolate.splrep(t[ext_maxima], x[ext_maxima])
    upper = interpolate.splev(t, tck)
    tck = interpolate.splrep(t[ext_minima], x[ext_minima])
    lower = interpolate.splev(t, tck)
    return upper, lower


def extr(x):

    m = x.shape[0]

    x1 = x[:m - 1]
    x2 = x[1:m]
    indzer = np.where(x1 * x2 < 0)[0]
    if np.any(x == 0):
        iz = np.where(x == 0)[0]
        indz = []
        if np.any(np.diff(iz) == 1):
            zer = x == 0
            dz = np.diff(np.r_[0, zer, 0])
            debz = np.where(dz == 1)[0]
            finz = np.where(dz == -1)[0] - 1
            indz = np.round((debz + finz) / 2)
        else:
            indz = iz
        indzer = np.sort(np.hstack([indzer, indz]))

    indmax = argrelmax(x)[0]
    indmin = argrelmin(x)[0]

    return indmin, indmax, indzer
