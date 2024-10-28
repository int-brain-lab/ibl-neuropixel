import datetime
from pathlib import Path

import scipy.signal

from one.api import ONE
from brainbox.io.spikeglx import Streamer
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import BrainRegions
from viewephys.gui import viewephys

from ibldsp import voltage, fourier
from neuropixel import trace_header


br = BrainRegions()
h = trace_header(version=1)
LFP_RESAMPLE_FACTOR = 10  # 250 Hz data

one = ONE(base_url="https://alyx.internationalbrainlab.org")
pid = "ce397420-3cd2-4a55-8fd1-5e28321981f4"
s0, sample_duration = (546, 30)
sr = Streamer(pid=pid, one=one, remove_cached=False, typ="lf")
tsel = slice(int(s0), int(s0) + int(sample_duration * sr.fs))
raw = sr[tsel, : -sr.nsync].T
destripe = voltage.destripe_lfp(
    raw, fs=sr.fs, neuropixel_version=1, channel_labels=True
)
destripe = scipy.signal.decimate(destripe, LFP_RESAMPLE_FACTOR, axis=1, ftype="fir")
fs_out = sr.fs / LFP_RESAMPLE_FACTOR
channels = SpikeSortingLoader(pid=pid, one=one).load_channels()

butter_kwargs = {"N": 3, "Wn": 2 / sr.fs * 2, "btype": "highpass"}
sos = scipy.signal.butter(**butter_kwargs, output="sos")
butter = scipy.signal.sosfiltfilt(sos, raw)
butter = fourier.fshift(butter, h["sample_shift"], axis=1)
butter = scipy.signal.decimate(butter, LFP_RESAMPLE_FACTOR, axis=1, ftype="fir")


## %%
eqcs = {}


csd = voltage.current_source_density(destripe, h)
eqcs["butter"] = viewephys(butter, fs=250, title="butter", channels=channels, br=br)
eqcs["destripe"] = viewephys(
    destripe, fs=250, title="destripe", channels=channels, br=br
)
eqcs["csd_butter"] = viewephys(
    voltage.current_source_density(butter, h) * 40**2,
    fs=250,
    title="csd_butter",
    channels=channels,
    br=br,
)
eqcs["csd_destripe"] = viewephys(
    voltage.current_source_density(destripe, h) * 40**2,
    fs=250,
    title="csd_destripe",
    channels=channels,
    br=br,
)

today = datetime.date.today().strftime("%Y-%m-%d")
out_path = Path("/home/ibladmin/Pictures")
for name, eqc in eqcs.items():
    eqc.viewBox_seismic.setXRange(8000, 10000)
    eqc.ctrl.set_gain(42)
    eqc.resize(1960, 1000)
    eqc.grab().save(str(out_path.joinpath(f"{today}_{name}.png")))

## %%
# from gpcsd.gpcsd2d import GPCSD2D
# from gpcsd.covariances import GPCSD2DSpatialCov, GPCSD2DSpatialCovSE, GPCSDTemporalCovMatern, GPCSDTemporalCovSE
# from gpcsd.priors import GPCSDInvGammaPrior, GPCSDHalfNormalPrior
# def gpcsd(lfp, xy, t):
#     R_prior = GPCSDInvGammaPrior()
#     R_prior.set_params(50, 300)
#     ellSEprior = GPCSDInvGammaPrior()
#     ellSEprior.set_params(20, 200)
#     temporal_cov_SE = GPCSDTemporalCovSE(t, ell_prior=ellSEprior)
#     ellMprior = GPCSDInvGammaPrior()
#     ellMprior.set_params(1, 20)
#     temporal_cov_M = GPCSDTemporalCovMatern(t, ell_prior=ellMprior)
#     gpcsd_model = GPCSD2D(lfp, xy, t, R_prior=R_prior,
#                           temporal_cov_list=[temporal_cov_SE, temporal_cov_M],
#                           eps=1, ngl1=30, ngl2=120,
#                           a1=np.min(xy[:, 0]) - 16, b1=np.max(xy[:, 0]) + 16,
#                           a2=np.min(xy[:, 1]) - 100, b2=np.max(xy[:, 1]) + 100)
#     print(gpcsd_model)
