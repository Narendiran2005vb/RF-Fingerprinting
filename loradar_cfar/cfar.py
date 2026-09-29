"""
Cell-Averaging CFAR.

The implementation works in LINEAR POWER, not dB.

For 2-D CA-CFAR, a rectangular neighborhood is used:
    [training + guard + CUT + guard + training]

The local noise estimate is obtained with summed-area style filtering
via scipy.ndimage.uniform_filter, so it is much faster than nested
Python loops.
"""

import numpy as np
from scipy.ndimage import uniform_filter


def ca_threshold_scale(pfa, n_training):
    """CA-CFAR alpha for exponentially distributed power noise."""
    if not 0 < pfa < 1:
        raise ValueError("pfa must be between 0 and 1.")
    if n_training <= 0:
        raise ValueError("n_training must be positive.")

    return n_training * (pfa ** (-1.0 / n_training) - 1.0)


def ca_cfar_2d(
    power,
    training_freq=20,
    guard_freq=4,
    training_time=8,
    guard_time=2,
    pfa=1e-5,
):
    """
    2-D Cell-Averaging CFAR.

    Parameters
    ----------
    power : 2-D ndarray
        Linear power, shape = [frequency_bins, time_frames].

    training_freq : int
        Training cells on EACH side in frequency.

    guard_freq : int
        Guard cells on EACH side in frequency.

    training_time : int
        Training cells on EACH side in time.

    guard_time : int
        Guard cells on EACH side in time.

    pfa : float
        Desired probability of false alarm.

    Returns
    -------
    detection : bool ndarray
        True at detected CUTs.

    threshold : ndarray
        Local CA-CFAR threshold in linear power.

    noise_estimate : ndarray
        Estimated local noise power.

    alpha : float
        CA-CFAR threshold multiplier.

    training_count : int
        Number of training cells used.
    """
    P = np.asarray(power, dtype=np.float64)

    if P.ndim != 2:
        raise ValueError("power must be a 2-D array.")
    if np.any(P < 0):
        raise ValueError("Power must be non-negative.")

    tf = int(training_freq)
    gf = int(guard_freq)
    tt = int(training_time)
    gt = int(guard_time)

    if min(tf, gf, tt, gt) < 0:
        raise ValueError("Cell counts cannot be negative.")

    # Outer window includes training + guard + CUT.
    outer_f = 2 * (tf + gf) + 1
    outer_t = 2 * (tt + gt) + 1

    # Inner window is CUT + guard cells.
    inner_f = 2 * gf + 1
    inner_t = 2 * gt + 1

    # Uniform filters give local means. Convert them to sums.
    outer_mean = uniform_filter(P, size=(outer_f, outer_t),
                                mode="constant", cval=0.0)
    inner_mean = uniform_filter(P, size=(inner_f, inner_t),
                                mode="constant", cval=0.0)

    outer_sum = outer_mean * (outer_f * outer_t)
    inner_sum = inner_mean * (inner_f * inner_t)

    n_training = outer_f * outer_t - inner_f * inner_t

    noise_estimate = (outer_sum - inner_sum) / n_training

    alpha = ca_threshold_scale(pfa, n_training)
    threshold = alpha * noise_estimate

    detection = P > threshold

    # Border cells do not have a complete training window.
    hf = tf + gf
    ht = tt + gt

    valid = np.zeros_like(P, dtype=bool)
    if P.shape[0] > 2 * hf and P.shape[1] > 2 * ht:
        valid[hf:-hf, ht:-ht] = True

    detection &= valid
    threshold[~valid] = np.nan
    noise_estimate[~valid] = np.nan

    return detection, threshold, noise_estimate, alpha, n_training


def ca_cfar_1d(
    power,
    training_cells=20,
    guard_cells=4,
    pfa=1e-5,
):
    """1-D CA-CFAR for one frequency spectrum."""
    P = np.asarray(power, dtype=np.float64).ravel()

    N = 2 * (training_cells + guard_cells) + 1
    G = 2 * guard_cells + 1

    outer = uniform_filter(P, size=N, mode="constant", cval=0.0) * N
    inner = uniform_filter(P, size=G, mode="constant", cval=0.0) * G

    n_training = N - G
    noise = (outer - inner) / n_training

    alpha = ca_threshold_scale(pfa, n_training)
    threshold = alpha * noise

    detection = P > threshold

    h = training_cells + guard_cells
    valid = np.zeros_like(P, dtype=bool)
    if len(P) > 2 * h:
        valid[h:-h] = True

    detection &= valid
    threshold[~valid] = np.nan
    noise[~valid] = np.nan

    return detection, threshold, noise, alpha, n_training
