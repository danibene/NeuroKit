# -*- coding: utf-8 -*-
import numpy as np

from ..misc.report import get_default_args
from .ecg_clean import ecg_clean
from .ecg_findpeaks import ecg_findpeaks


def ecg_methods(
    sampling_rate=1000,
    method="neurokit",
    method_cleaning="default",
    method_peaks="default",
    **kwargs,
):
    """**PPG Preprocessing Methods**

    This function analyzes and specifies the methods used in the preprocessing, and create a
    textual description of the methods used. It is used by :func:`ppg_process()` to dispatch the
    correct methods to each subroutine of the pipeline and :func:`ppg_report()` to create a
    preprocessing report.

    Parameters
    ----------
    sampling_rate : int
        The sampling frequency of the raw PPG signal (in Hz, i.e., samples/second).
    method : str
        The method used for cleaning and peak finding if ``"method_cleaning"``
        and ``"method_peaks"`` are set to ``"default"``. Defaults to ``"neurokit"``.
    method_cleaning: str
        The method used to clean the raw PPG signal. If ``"default"``,
        will be set to the value of ``"method"``. Defaults to ``"default"``.
        For more information, see the ``"method"`` argument
        of :func:`.ecg_clean`.
    method_peaks: str
        The method used to find peaks. If ``"default"``,
        will be set to the value of ``"method"``. Defaults to ``"default"``.
        For more information, see the ``"method"`` argument
        of :func:`.ecg_findpeaks`.
    **kwargs
        Other arguments to be passed to :func:`.ecg_clean` and
        :func:`.ecg_findpeaks`.

    Returns
    -------
    report_info : dict
        A dictionary containing the keyword arguments passed to the cleaning
        and peak finding functions, text describing the methods, and the corresponding
        references.

    See Also
    --------
    ecg_process, ecg_clean, ecg_findpeaks

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      methods = nk.ecg_methods(sampling_rate=100, method="neurokit", method_cleaning="pantompkins1985")
      print(methods["text_cleaning"])
      print(methods["references"][0])

    """
    # Sanitize inputs
    if method_cleaning == "default":
        method_cleaning = method
    if method_peaks == "default":
        method_peaks = method

    # Create dictionary with all inputs
    report_info = {
        "sampling_rate": sampling_rate,
        "method": method,
        "method_cleaning": method_cleaning,
        "method_peaks": method_peaks,
        **kwargs,
    }

    # Get arguments to be passed to cleaning and peak finding functions

    defaults_cleaning = get_default_args(ecg_clean)
    defaults_peaks = get_default_args(ecg_findpeaks)

    kwargs_cleaning = {}
    for key in defaults_cleaning.keys():
        if key not in ["sampling_rate", "method"]:
            # if arguments have not been specified by user,
            # set them to the defaults
            if key not in report_info.keys():
                report_info[key] = defaults_cleaning[key]
            elif report_info[key] != defaults_cleaning[key]:
                kwargs_cleaning[key] = report_info[key]
    kwargs_peaks = {}

    for key in defaults_peaks.keys():
        if key not in ["sampling_rate", "method"]:
            # if arguments have not been specified by user,
            # set them to the defaults
            if key not in report_info.keys():
                report_info[key] = defaults_peaks[key]
            elif report_info[key] != defaults_peaks[key]:
                kwargs_peaks[key] = report_info[key]

    # Save keyword arguments in dictionary
    report_info["kwargs_cleaning"] = kwargs_cleaning
    report_info["kwargs_peaks"] = kwargs_peaks

    # Initialize refs list
    refs = []

    # 1. Cleaning
    # ------------
    report_info["text_cleaning"] = f"The raw signal, sampled at {sampling_rate} Hz, "
    if method_cleaning in ["neurokit"]:
        report_info["text_cleaning"] = (
            report_info["text_cleaning"]
            + "was preprocessed using a bandpass filter ([0.5 - 8 Hz], Butterworth 3rd order"
            + "; high-pass butterworth filter (order = 5), followed by powerline filtering "
            + "following "
        )
        refs.append(
            "Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection"
            + " in Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical"
            + " Conditions. PLoS ONE 8(10): e76585. doi:10.1371/journal.pone.0076585."
        )
    elif method_cleaning in ["nabian2018"]:
        if report_info["heart_rate"] is None:
            cutoff = "of 40 Hz"
        else:
            cutoff = f'based on the heart rate of {report_info["heart_rate"]} bpm'

        report_info["text_cleaning"] = (
            report_info["text_cleaning"]
            + "was preprocessed using a lowpass filter (with a cutoff frequency "
            + f"{cutoff}, butterworth 2nd order; following Nabian et al., 2018)."
        )

        refs.append(
            "Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., & Ostadabbas, S."
            + " (2018). An open-source feature extraction tool for the analysis of peripheral "
            + "physiological data. IEEE Journal of Translational Engineering in Health and Medicine"
            + ", 6, 1-11."
        )
    elif method_cleaning is None or method_cleaning.lower() == "none":
        report_info["text_cleaning"] = (
            report_info["text_cleaning"]
            + "was directly used for peak detection without preprocessing."
        )
    else:
        # just in case more methods are added
        report_info["text_cleaning"] = "was cleaned following the " + method + " method."

    # 2. Peaks
    # ----------
    if method_peaks in ["elgendi"]:
        report_info[
            "text_peaks"
        ] = "The peak detection was carried out using the method described in Elgendi et al. (2013)."
        refs.append(
            "Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection"
            + " in Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical"
            + " Conditions. PLoS ONE 8(10): e76585. doi:10.1371/journal.pone.0076585."
        )
    report_info["references"] = list(np.unique(refs))

    # Print text
    for key in ["text_cleaning", "text_peaks", "references"]:
        if isinstance(report_info[key], list):
            for s in report_info[key]:
                print(s)
        else:
            print(report_info[key])
        print("")

    return report_info
