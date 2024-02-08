import numpy as np
from ibldsp.fourier import fshift


def generate_waveform(spike=None, sxy=None, wxy=None, fs=30000, vertical_velocity_mps=3):
    """
    Generate a waveform from a spike and a set of coordinates
    :param spike: the single trace spike waveform
    :param sxy: 3 elements np.array containing coordinates of the spike
    :param wxy: ntraces by 3 np.array containing the generated traces coordinates
    :param fs: sampling frequency
    :param vertical_velocity_mps: vertical velocity of the spike in m/s
    :return: the generated waveform ns by ntraces
    """
    # spike coordinates
    sxy = np.array([43., 1940., 0.]) if sxy is None else sxy
    # generated traces coordinates
    if wxy is None:
        wxy = np.array([
            [43., 1740., 0.],
            [59., 1760., 0.],
            [27., 1760., 0.],
            [43., 1780., 0.],
            [11., 1780., 0.],
            [59., 1800., 0.],
            [27., 1800., 0.],
            [43., 1820., 0.],
            [11., 1820., 0.],
            [59., 1840., 0.],
            [27., 1840., 0.],
            [43., 1860., 0.],
            [11., 1860., 0.],
            [59., 1880., 0.],
            [27., 1880., 0.],
            [43., 1900., 0.],
            [11., 1900., 0.],
            [59., 1920., 0.],
            [27., 1920., 0.],
            [43., 1940., 0.],
            [11., 1940., 0.],
            [59., 1960., 0.],
            [27., 1960., 0.],
            [43., 1980., 0.],
            [11., 1980., 0.],
            [59., 2000., 0.],
            [27., 2000., 0.],
            [43., 2020., 0.],
            [11., 2020., 0.],
            [59., 2040., 0.],
            [27., 2040., 0.],
            [43., 2060., 0.],
            [11., 2060., 0.],
            [59., 2080., 0.],
            [27., 2080., 0.],
            [43., 2100., 0.],
            [11., 2100., 0.],
            [59., 2120., 0.],
            [27., 2120., 0.],
            [43., 2140., 0.]])

    # spike waveform
    if spike is None:
        spike = np.array([
            1.39729483e-02, 8.35108757e-03, 1.19482297e-02, 8.70722998e-03,
            7.36843236e-03, -1.13049131e-02, -1.55959455e-02, -2.61027571e-02,
            -2.17338502e-02, -4.38232124e-02, -5.93647808e-02, -7.47340098e-02,
            -9.37892646e-02, -1.18461445e-01, -1.39234722e-01, -1.30095094e-01,
            -1.10826254e-01, -8.66197199e-02, -9.47982967e-02, -8.71859193e-02,
            3.44982445e-02, 1.14850193e-01, 1.52717680e-01, 1.71819285e-01,
            1.89162567e-01, 1.66016370e-01, 1.77806437e-01, 1.37514159e-01,
            1.76143244e-01, 1.78440601e-01, 1.70397654e-01, 1.42554864e-01,
            1.80504188e-01, 2.94411600e-01, 4.22272682e-01, 4.51008588e-01,
            3.66263747e-01, 1.88719124e-01, -2.11601019e-01, -9.27919567e-01,
            -1.91966367e+00, -2.83236146e+00, -3.26066566e+00, -2.75487232e+00,
            -1.70523679e+00, -5.07467031e-01, 4.15351689e-01, 1.14206159e+00,
            1.42395592e+00, 1.47938919e+00, 1.48601341e+00, 1.39127147e+00,
            1.34073710e+00, 1.21446955e+00, 1.07075334e+00, 9.82654214e-01,
            9.20302153e-01, 7.62362599e-01, 6.83112204e-01, 5.65638483e-01,
            4.95534867e-01, 4.45296884e-01, 3.31838697e-01, 1.83607757e-01,
            1.10464394e-01, 7.94331133e-02, 5.71388900e-02, -1.53203905e-02,
            -1.13413632e-01, -2.63738900e-01, -2.96958297e-01, -2.99410105e-01,
            -3.27556312e-01, -4.34036553e-01, -5.66587210e-01, -5.50931454e-01,
            -4.83926237e-01, -3.96159559e-01, -3.59628379e-01, -2.93378174e-01,
            -2.23388135e-01, -1.75207376e-01, -1.44064426e-01, -8.60679895e-02,
            -5.16730249e-02, -6.04236871e-02, -7.13021904e-02, -5.77894375e-02,
            -5.49767427e-02, -5.17059378e-02, -3.11024077e-02, -2.73740329e-02,
            -3.09202522e-02, -3.67176980e-02, -3.99643928e-02, -5.43142855e-02,
            -6.30898550e-02, -6.07964136e-02, -4.08532396e-02, -2.44005471e-02,
            -3.96704227e-02, -1.90648790e-02, 9.41569358e-03, 3.47820818e-02,
            4.08176184e-02, 3.42404768e-02, 3.01315673e-02, 2.90315691e-02,
            2.64853500e-02, 2.18018480e-02, 1.19765718e-02, 4.67543490e-03,
            2.74471682e-03, -2.62711023e-04, 1.84994331e-03, 6.98080519e-03,
            1.11559704e-02, 1.33141074e-02, 1.58480220e-02, 1.66855101e-02,
            1.60783399e-02], dtype=np.float32)

    r = np.sqrt(np.sum(np.square(sxy - wxy), axis=1))
    sample_shift = (wxy[:, 1] - np.mean(wxy[:, 1])) / 1e6 * vertical_velocity_mps * fs
    # shperical divergence
    wav = (spike * 1 / (r[..., np.newaxis] + 50)**2)
    wav = fshift(wav, sample_shift, axis=-1).T
    return wav
