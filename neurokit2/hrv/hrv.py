# -*- coding: utf-8 -*-
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ..stats import summary_plot
from .hrv_frequency import _hrv_frequency_show, hrv_frequency
from .hrv_nonlinear import _hrv_nonlinear_show, hrv_nonlinear
from .hrv_rsa import hrv_rsa
from .hrv_time import hrv_time
from .hrv_utils import _hrv_get_rri, _hrv_sanitize_input
from ..signal import signal_interpolate


def hrv(data, rri_time=None, data_format="peaks", sampling_rate=1000, show=False, check_successive=True, **kwargs):
    """**Heart Rate Variability (HRV)**

    This function computes all HRV indices available in NeuroKit. It is essentially a convenience
    function that aggregates results from the :func:`time domain <hrv_time>`, :func:`frequency
    domain <hrv_frequency>`, and :func:`non-linear domain <hrv_nonlinear>`.

    .. tip::

        We strongly recommend checking our open-access paper `Pham et al. (2021)
        <https://doi.org/10.3390/s21123998>`_ on HRV indices for more information.

    Parameters
    ----------
    data : dict, list or ndarray
        If data format is peaks, Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur.
        Can be a list of indices or the output(s) of other functions such as ecg_peaks,
        ppg_peaks, ecg_process or bio_process.
        If data format is R-R intervals, list or ndarray of R-R intervals.
    rri_time : list or ndarray, optional
        Time points corresponding to R-R intervals, in seconds.
    data_format : str, optional
        If "peaks", the R-R intervals are computed with hrv_get_rri.
        If "rri", the input is assumed to already be R-R intervals and they are not computed.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur. Should be at
        least twice as high as the highest frequency in vhf. By default 1000.
    show : bool, optional
        If ``True``, returns the plots that are generates for each of the domains.
    check_successive: bool, optional
            If True, will remove non-successive differences based on whether the R-R intervals match the corresponding
            timepoints or if there are NaN values (only for HRV indices that assume successive differences e.g. RMSSD).
            Should be set to False if R-R intervals are interpolated, as the R-R intervals will then not match the
            timepoints.

    Returns
    -------
    DataFrame
        Contains HRV indices in a DataFrame. If RSP data was provided (e.g., output of
        :func:`bio_process`), RSA indices will also be included.

    See Also
    --------
    hrv_time, hrv_frequency, hrv_nonlinear, hrv_rsa, .ecg_peaks, ppg_peaks,

    Examples
    --------
    **Example 1**: Only using a list of R-peaks locations

    .. ipython:: python

      import neurokit2 as nk
      import matplotlib.pyplot as plt
      plt.rc('font', size=8)

      # Download data
      data = nk.data("bio_resting_5min_100hz")

      # Clean signal and Find peaks
      ecg_cleaned = nk.ecg_clean(data["ECG"], sampling_rate=100)
      peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=100, correct_artifacts=True)

      # Compute HRV indices
      @savefig p_hrv1.png scale=100%
      hrv_indices = nk.hrv(peaks, sampling_rate=100, show=True)
      @suppress
      plt.close()

    **Example 2**: Compute HRV directly from processed data

    .. ipython:: python

      # Download data
      data = nk.data("bio_resting_5min_100hz")

      # Process
      signals, info = nk.bio_process(data, sampling_rate=100)

      # Get HRV
      nk.hrv(signals, sampling_rate=100)


    References
    ----------
    * Pham, T., Lau, Z. J., Chen, S. H. A., & Makowski, D. (2021). Heart Rate Variability in
      Psychology: A Review of HRV Indices and an Analysis Tutorial. Sensors, 21(12), 3998.
      https://doi.org/10.3390/s21123998
    * Stein, P. K. (2002). Assessing heart rate variability from real-world Holter reports. Cardiac
      electrophysiology review, 6(3), 239-244.
    * Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and
      norms. Frontiers in public health, 5, 258.

    """
    # Get indices
    out = []  # initialize empty container

    # Gather indices
    out.append(hrv_time(data, rri_time=rri_time, data_format=data_format, sampling_rate=sampling_rate,
                        check_successive=check_successive))
    out.append(hrv_frequency(data, rri_time=rri_time, data_format=data_format, sampling_rate=sampling_rate))
    out.append(hrv_nonlinear(data, rri_time=rri_time, data_format=data_format, sampling_rate=sampling_rate,
                             check_successive=check_successive))

    # Compute RSA if rsp data is available
    if isinstance(data, pd.DataFrame):
        if ("RSP_Phase" in data.columns) and ("RSP_Phase_Completion" in data.columns):
            rsp_signals = data[["RSP_Phase", "RSP_Phase_Completion"]]
            rsa = hrv_rsa(data, rsp_signals, sampling_rate=sampling_rate)
            out.append(pd.DataFrame([rsa]))

    out = pd.concat(out, axis=1)

    # Plot
    if show:
        if data_format == "peaks":
            if isinstance(data, dict):
                data = data["ECG_R_Peaks"]

        # Indices for plotting
        out_plot = out.copy(deep=False)

        _hrv_plot(data, 
                  out=out_plot, 
                  rri_time=rri_time, 
                  data_format=data_format, 
                  sampling_rate=sampling_rate,
                  check_successive=check_successive, 
                  **kwargs)

    return out


def _hrv_plot(data, out, rri_time=None, data_format="peaks", sampling_rate=1000, check_successive=True, **kwargs):

    fig = plt.figure(constrained_layout=False)
    spec = gs.GridSpec(ncols=2, nrows=2, height_ratios=[1, 1], width_ratios=[1, 1])

    # Arrange grids
    ax_distrib = fig.add_subplot(spec[0, :-1])
    ax_distrib.set_xlabel("R-R intervals (ms)")
    ax_distrib.set_title("Distribution of R-R intervals")

    ax_psd = fig.add_subplot(spec[1, :-1])

    spec_within = gs.GridSpecFromSubplotSpec(
        4, 4, subplot_spec=spec[:, -1], wspace=0.025, hspace=0.05
    )
    ax_poincare = fig.add_subplot(spec_within[1:4, 0:3])
    ax_marg_x = fig.add_subplot(spec_within[0, 0:3])
    ax_marg_x.set_title("Poincaré Plot")
    ax_marg_y = fig.add_subplot(spec_within[1:4, 3])

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # Format data as R-R intervals without NaN values
    if data_format == "peaks":
        peaks = data
        # Sanitize input
        peaks = _hrv_sanitize_input(peaks)
        # Compute R-R intervals (also referred to as NN) in milliseconds
        rri, _ = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=False)
    else:
        rri = np.array(data)
        if rri_time is None:
            # Compute the timestamps of the R-R intervals in seconds
            rri_time = np.nancumsum(rri / 1000)

        # Remove NaN R-R intervals, if any
        rri_time = rri_time[~np.isnan(rri)]
        rri = rri[~np.isnan(rri)]

    # Distribution of RR intervals
    ax_distrib = summary_plot(rri, ax=ax_distrib, **kwargs)

    # Poincare plot
    out.columns = [col.replace("HRV_", "") for col in out.columns]
    _hrv_nonlinear_show(rri, out, ax=ax_poincare, ax_marg_x=ax_marg_x, ax_marg_y=ax_marg_y,
                        rri_time=rri_time, check_successive=check_successive)

    # PSD plot
    if data_format == "peaks":
        rri, sampling_rate = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=True)
    else:
        # Sanitize minimum sampling rate for interpolation to 10 Hz, so that output is the same as _hrv_get_rri
        sampling_rate = max(sampling_rate, 10)

        x_new = np.arange(np.floor(sampling_rate * rri_time[0]), np.ceil(sampling_rate * rri_time[-1])) / sampling_rate
        rri = signal_interpolate(
            rri_time,
            rri,
            x_new=x_new,
            **kwargs
        )

    frequency_bands = out[["ULF", "VLF", "LF", "HF", "VHF"]]
    _hrv_frequency_show(rri, frequency_bands, sampling_rate=sampling_rate, ax=ax_psd)
