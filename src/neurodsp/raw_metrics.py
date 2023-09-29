from neurodsp.voltage import destripe, destripe_lfp, detect_bad_channels
from neurodsp.utils import rms
import spikeglx
import pandas as pd
import scipy
from tqdm import trange


def raw_data_features(ap_cbin, lf_cbin, t_start, t_end):
    """
    Create a QC table with the following features for each channel for
    each snippets:

    - RMS of raw, butterworth, destriped, and residual (butterworth - destriped)
    for both bands (returned in uV).

    - DC offset of raw data for both bands (returned in uV).

    - Channel labels estimated from both AP and LF bands separately.

    - Channel detection features (LF and HF cross-correlation and HF PSD)
    estimated from both AP and LF bands separately.

    AP band data is filtered with a 300Hz high-pass butterworth for
    the BTR and DST features. LF band is filtered with a 2-200Hz band-pass
    butterworth. Snippets of 12x the length at the same starting times are
    taken from LF to account for different sampling rates.

    Returns a Pandas Multiindex dataframe with snippet and channel ids.

    :param ap_cbin: Path to AP band cbin.
    :param lf_cbin: Path ot LF band cbin.
    :param t_start: Starting times of snippets in seconds.
    :param t_end: ending times of snippets in seconds.
    """
    assert len(t_start) == len(t_end), "mismatched snippet start and end"
    num_snippets = len(t_start)

    sr_ap = spikeglx.Reader(ap_cbin)
    sr_lf = spikeglx.Reader(lf_cbin)

    filter_ap = scipy.signal.butter(
        N=3, Wn=300 / sr_ap.fs * 2, btype="highpass", output="sos"
    )
    filter_lf = scipy.signal.butter(
        N=3, Wn=[2 / sr_lf.fs * 2, 200 / sr_ap.fs * 2], btype="bandpass", output="sos"
    )

    dfs = {}

    print(f"Extracting features over {num_snippets} snippets")
    for i in trange(num_snippets):
        t0 = t_start[i]
        t1 = min(t_end[i], sr_ap.rl)
        df = compute_raw_features_snippet(sr_ap, sr_lf, t0, t1, filter_ap, filter_lf)
        dfs[i] = df

    out_df = pd.concat(dfs)
    out_df.index.rename(["snippet_id", "channel_id"], inplace=True)

    return out_df


def compute_raw_features_snippet(sr_ap, sr_lf, t0, t1, filter_ap=None, filter_lf=None):
    """
    Compute raw data feature table on one snippet of AP / LF data.
    Returns
    """
    # conversion from V to uV
    factor = 1.0e6

    if filter_ap is None:
        filter_ap = scipy.signal.butter(
            N=3, Wn=300 / sr_ap.fs * 2, btype="highpass", output="sos"
        )
    if filter_lf is None:
        filter_lf = scipy.signal.butter(
            N=3,
            Wn=[2 / sr_lf.fs * 2, 200 / sr_lf.fs * 2],
            btype="bandpass",
            output="sos",
        )
    filters = {"ap": filter_ap, "lf": filter_lf}
    sglx = {"ap": sr_ap, "lf": sr_lf}
    detect_kwargs = {
        "ap": {"fs": sr_ap.fs, "psd_hf_threshold": None},
        "lf": {"fs": sr_lf.fs, "psd_hf_threshold": 1.4},
    }
    destripe_fn = {"ap": destripe, "lf": destripe_lfp}

    # sample 12x length for AP
    t1_ap = t1
    t_length = t1 - t0
    t_length *= 12
    t1_lf = min(t0 + t_length, sr_lf.rl)
    t1s = {"ap": t1_ap, "lf": t1_lf}

    data = {}

    for band in ["ap", "lf"]:
        sr = sglx[band]
        sl = slice(int(sr.fs * t0), int(sr.fs * t1s[band]))
        raw = sr[sl, : -sr.nsync].T
        channel_labels, xfeats = detect_bad_channels(raw, **detect_kwargs[band])
        butter = scipy.signal.sosfiltfilt(filters[band], raw)
        destriped = destripe_fn[band](raw, fs=sr.fs, channel_labels=channel_labels)
        # get raw rms for free
        raw_rms = xfeats["rms_raw"]
        butter_rms = rms(butter)
        destripe_rms = rms(destriped)
        residual_rms = rms(butter - destriped)

        data[f"{band}_raw_rms"] = raw_rms * factor
        data[f"{band}_butter_rms"] = butter_rms * factor
        data[f"{band}_destripe_rms"] = destripe_rms * factor
        data[f"{band}_residual_rms"] = residual_rms * factor
        data[f"{band}_channel_labels"] = channel_labels
        # channel detect features
        data[f"{band}_xcor_hf"] = xfeats["xcor_hf"]
        data[f"{band}_xcor_lf"] = xfeats["xcor_lf"]
        data[f"{band}_psd_hf"] = xfeats["psd_hf"]

    return pd.DataFrame(data)
