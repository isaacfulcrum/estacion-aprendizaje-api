import numpy as np
from scipy import stats

from enum import Enum


class PerformanceTrend(Enum):
    STEADY_IMPROVEMENT = 1
    SLIGHT_DECLINE = 2
    STABLE_WITH_RECOVERY = 3
    CONSISTENTLY_HIGH = 4
    CONSISTENT_DECLINE = 5
    SLIGHT_IMPROVEMENT = 6
    STEADY_DECLINE = 7
    CONSISTENTLY_AVERAGE = 8
    FLUCTUATING_BUT_STABLE = 9
    CONSISTENTLY_LOW = 10


def get_trend(scores):
    # Create an array of time points
    times = np.arange(len(scores))
    # Fit a linear regression model
    slope, intercept, r_value, p_value, std_err = stats.linregress(times, scores)
    # Calculate the latest score
    latest_score = scores[-1]
    # Set thresholds (these values may need to be adjusted)
    high_threshold = 0.8
    low_threshold = 0.2
    improvement_threshold = 0.005
    # Assign a trend category based on the slope and the latest score
    if slope > improvement_threshold and latest_score >= high_threshold:
        return PerformanceTrend.STEADY_IMPROVEMENT
    elif slope < -improvement_threshold and latest_score <= low_threshold:
        return PerformanceTrend.CONSISTENT_DECLINE
    elif slope > improvement_threshold:
        return PerformanceTrend.SLIGHT_IMPROVEMENT
    elif slope < -improvement_threshold:
        return PerformanceTrend.SLIGHT_DECLINE
    elif latest_score >= high_threshold:
        return PerformanceTrend.CONSISTENTLY_HIGH
    elif latest_score <= low_threshold:
        return PerformanceTrend.CONSISTENTLY_LOW
    else:
        return PerformanceTrend.CONSISTENTLY_AVERAGE
