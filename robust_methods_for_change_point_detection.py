import numpy as np
from astropy.stats import biweight_midvariance
from scipy import stats
from typing import Tuple

def quantile_conditional_variance(
    data: np.array, j: int, k: int, a: float = 0.05, b: float = 0.05
) -> float:
    """
    Calculate quantile conditional variance.

    The definition available in the following paper:
    Pitera, M., Chechkin, A., & Wyłomańska, A. (2022).
    Goodness-of-fit test for α-stable distribution based on the quantile conditional variance statistics.
    Statistical Methods & Applications, 31(2), 387-424.

    :param data: one-dimensional data in the form of a vector
    :param j: the number of observations since we analyze data
    :param k: the number of the last observation from data that we consider
    :param a: the quantile that we considered (low level of cut our data)
    :param b: the quantile that we considered (high level of cut our data)

    :return Cj: QCV statistic
    """

    X = np.sort(data[j:k])
    n = len(X)

    # Calculate the indices na and nb directly
    na, nb = int(round(n * a) + 1), n - int(round(n * b))

    # Use slicing to get the relevant portion of X and calculate the mean
    X_subset = X[na:nb]
    mean_X_subset = np.mean(X_subset)

    # Calculate the sum of squares of deviations from the mean
    sum_squares = np.sum((X_subset - mean_X_subset) ** 2)

    # Calculate the robust estimator
    Cj = sum_squares / (nb - na)
    return Cj


def statistics_ICSS(data: np.array, n: int) -> float:
    """
    Calculate value of statistic S(n) using formula (2).

    :param data: two-regimes data with changing variance.
    :param n: selected index of data observed

    :return: the value of statistic S(n)
    """
    N = len(data)
    Cn = sum(np.array(data[0:n]) ** 2)
    CN = sum(data**2)
    return Cn / CN - n / N


def baseline_ICSS(data: np.array) -> int:
    """
    Calculate single change point for gaussian data with changing variance.

    The methodology comes from (version for a single change point detection):
    Inclan, C., & Tiao, G. C. (1994).
    Use of cumulative sums of squares for retrospective detection of changes of variance.
    Journal of the American Statistical Association, 89(427), 913-923.

    :param data: two-regimes data with changing variance.

    :return change_point: obtained change point.
    """
    N = len(data)
    statistics = [np.abs(statistics_ICSS(data, k)) for k in range(N)]
    return statistics.index(max(statistics))


def statistics_OLS(C_j: np.array, j: int) -> float:
    """
    Calculate value of loss function using in the 'baseline_OLS' in the point j.

    :param C_j: two-regimes data with changing variance.
    :param j: index of observation

    :return: The value of the loss function using for change point detection.
    """
    N = len(C_j)
    x1 = np.array(range(0, j + 1))
    x2 = np.array(range(j + 1, N + 1))
    y_j1 = np.poly1d(np.polyfit(x1, C_j[0 : j + 1], 1))(x1)
    y_j2 = np.poly1d(np.polyfit(x2, C_j[j : N + 1], 1))(x2)
    return sum((C_j[0 : j + 1] - y_j1) ** 2) + sum((C_j[j : N + 1] - y_j2) ** 2)


def baseline_OLS(data: np.array) -> np.int64:
    """
    Calculate single change point for data with changing variance.

    The methodology comes from (version for a single change point detection):
    G. Janusz, G. Sikora, A. Wyłomańska,
    Regime variance testing - a quantile approach,
    Acta Physica Polonica B (2013)

    :param data: two-regimes data with changing variance.

    :return change_point: obtained change point.
    """
    C_n = np.cumsum(data**2)
    N = len(C_n)
    loss_function = np.arange(N - 2, dtype=np.int64)
    for n in range(1, N - 1):
        loss_function[n - 1] = float(statistics_OLS(C_n, n))
    change_point = list(loss_function).index(min(loss_function)) + 1
    return change_point


def examine_convexity(
    robust_Cn: np.array, Sn: np.array, N: int
) -> Tuple[float, float]:
    """
    Calculate extremum of a given function described by selected points from the array (statistics).

    Inputs:
    :param robust_Cn: the values of robust C(n)
    :param Sn: the values of S(n) given by formula (2) but using robust versions of C(n)
    :param N: data length
    otherwise use the robust estimator of variance given by formula (9).

    :return value: the extremum of the function described by vector Sn
    :return extremum_index: index of 'value'
    """
    extremum_index = list(Sn).index(max(np.abs(Sn))) + 2
    p = np.polyfit([2, N - 2], [robust_Cn[1], robust_Cn[-1]], 1)
    value = np.polyval(p, extremum_index)
    return value, extremum_index


def select_version_Dk(
    data: np.array, selected_statistics: str = "BMID"
) -> Tuple[np.array, np.array]:
    """
    Select proper way of calculation of S(n) given by formula (2).

    The procedure of selecting this "way of calculation" correspont to the subsection 3.3.

    :param data: two-regimes data with changing variance.
    :param selected_statistics: if 'BMID' then use robust estimator of variance given by formula (8),
    otherwise use the robust estimator of variance given by formula (9).

    :return: robust version of C(n) in the form of vector
    :return: vector of S(n) given by formula (2) that use robust C(n) instead of CSS
    """
    N = len(data)
    stat = np.zeros(N)
    for j in range(1, N - 1):
        if selected_statistics == "BMID":
            stat[j] = (
                (j-1) * biweight_midvariance(data[:j])
                + j * np.median(data[:j]) ** 2
            )
        else:
            if j < int(0.05 * N):
                stat[j] = 0
            else:
                stat[j] = (
                    (j-1) * quantile_conditional_variance(data, 0, j)
                    + j * np.median(data[:j]) ** 2
                )
    Sn = np.zeros(N)
    CT = stat[N - 2]
    for j in range(3, N - 3):
        Ck = stat[j]
        Sn[j] = abs(Ck / CT - j / N)

    y1, L = examine_convexity(stat, Sn, N)
    M = stat[L]

    if y1 < M:
        BMID = stat
        Sn = np.zeros(N)
        CT = stat[N - 2]
        for j in range(2, N - 2):
            Ck = stat[j]
            Sn[j] = abs(Ck / CT - j / N)
        dk = Sn
    else:
        stat2 = np.zeros(N)
        for j in range(2, N - 1):
            if selected_statistics == "BMID":
                stat2[j] = (
                    (j-1) * biweight_midvariance(data[N - j :])
                    + j * np.median(data[N - j :]) ** 2
                )
            else:
                stat2[j] = (
                    (j-1) * quantile_conditional_variance(data, N - j, N)
                    + j * np.median(data[N - j :]) ** 2
                )
        stat2 = np.flip(stat2)

        if np.sum(stat2 > stat2[2]) / (N - 4) > 0.15:
            Sn2 = np.zeros(N)
            CT = stat2[N - 2]
            for j in range(2, N - 2):
                Ck = stat2[j]
                Sn2[j] = abs(Ck / CT - j / N)
            BMID = stat2
            dk = Sn2
        else:
            BMID = stat
            dk = Sn

    return BMID, dk


def robust_OLS(data: np.array, selected_statistics: str = "BMID") -> int:
    """
    Calculate single change point for data with changing scale.

    It is developed version of the methodology described here:
    G. Janusz, G. Sikora, A. Wyłomańska,
    Regime variance testing - a quantile approach,
    Acta Physica Polonica B (2013)

    :param data: two-regimes data with changing variance.
    :param selected_statistics: if 'BMID' then use robust estimator of variance given by formula (8),
    otherwise use the robust estimator of variance given by formula (9).

    :return change_point: obtained change point.
    """

    N = len(data)
    stat, _ = select_version_Dk(data, selected_statistics)
    loss_function = []
    for K in range(1, N - 1):
        loss_function.append(statistics_OLS(stat, K))
    change_point = list(loss_function).index(min(loss_function)) + 1
    return change_point


def robust_ICSS(data: np.array, selected_statistics: str = "BMID") -> float:
    """
    Calculate single change point for data with changing scale.

    It is developed version of the methodology described here:
    Inclan, C., & Tiao, G. C. (1994).
    Use of cumulative sums of squares for retrospective detection of changes of variance.
    Journal of the American Statistical Association, 89(427), 913-923.

    :param data: two-regimes data with changing variance.
    :param selected_statistics: if 'BMID' then use robust estimator of variance given by formula (8),
    otherwise use the robust estimator of variance given by formula (9).

    :return change_point: obtained change point via robust version of ICSS.
    """
    _, Sn = select_version_Dk(data, selected_statistics)
    Sn = list(np.abs(Sn))
    change_point = Sn.index(max(np.abs(Sn))) + 1
    return change_point


def change_points_detection(data: np.array, n1: int) -> dict:
    """
    Return the dictionary with true change point taken as n1 and segmentation results.

    The segmentation results are obtained for:
    - baseline ICSS,
    - robust ICSS with the BMID taken as a robust estimator of scale
    - robust ICSS with the QCV taken as a robust estimator of scale
    - baseline OLS,
    - robust OLS with the BMID taken as a robust estimator of scale
    - robust OLS with the QCV taken as a robust estimator of scale

    :param data: a vector with data with change in scale
    :param n1: true change point (i.e., the length of the first segment)

    :return change_points: dictionary with a true change point and the segmentation results
    """
    change_points = {"true": n1}
    change_points["ICSS_baseline"] = baseline_ICSS(data)
    change_points["ICSS_BMID"] = robust_ICSS(data, selected_statistics="BMID")
    change_points["ICSS_QCV"] = robust_ICSS(data, selected_statistics="QCV")
    change_points["OLS_baseline"] = baseline_OLS(data)
    change_points["OLS_BMID"] = robust_OLS(data, selected_statistics="BMID")
    change_points["OLS_QCV"] = robust_OLS(data, selected_statistics="QCV")
    return change_points


# Working example
n1, n2 = 500, 500
scale1 = 1
scale2 = 0.2
data = np.concatenate(
    [stats.cauchy.rvs(loc=0, scale=scale1, size=n1), stats.cauchy.rvs(loc=0, scale=scale2, size=n2)]
)
print(change_points_detection(data=data, n1=n1))
