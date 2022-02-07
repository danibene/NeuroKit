import numpy as np


def find_successive_intervals(intervals, intervals_time, thresh_diff_ms=2, n_diff=1):
    """Identify successive intervals

    Identification of intervals that are consecutive 
    (e.g. in case of missing data).

    Parameters
    ----------
    intervals : list or ndarray
        Intervals, e.g. breath-to-breath (BBI) or rpeak-to-rpeak (RRI)
    intervals_time : list or ndarray
        Time points corresponding to intervals, in seconds.
    thresh_diff_ms : int, float
        Threshold at which the difference between time points is considered to 
        be unequal to the interval, in milliseconds.
    n_diff: int
        The number of times values are differenced. 
        Can be used to check which values are valid for the n-th difference
        assuming successive intervals.

    Returns
    ----------
    successive_intervals: ndarray
        A list of True/False with True being the successive intervals.

    Example
    ----------
    >>> import neurokit2 as nk
    >>> rri = [400, 500, 700, 800, 900]
    >>> rri_time = [0.7,  1.2,  2.5, 3.3, 4.2]
    >>> successive_intervals = nk.find_successive_intervals(rri, rri_time)
    >>> successive_intervals
    array([ True, False,  True,  True])
    """
    
    # Convert to numpy array
    intervals_time = np.array(intervals_time)
    intervals = np.array(intervals)

    # Remove the timestamps of the NaN intervals (if any)
    intervals_time = intervals_time[~np.isnan(intervals)]
    # Remove the NaN intervals (if any)
    intervals = intervals[~np.isnan(intervals)]
      
    diff_intervals_time_ms = np.diff(intervals_time, n=n_diff)*1000
    
    abs_error_intervals_ref_time = abs(diff_intervals_time_ms - np.diff(intervals[1:], n=n_diff-1))
    
    successive_intervals = abs_error_intervals_ref_time <= thresh_diff_ms
      
    return np.array(successive_intervals)
