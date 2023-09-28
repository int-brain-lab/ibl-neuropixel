from neurodsp.voltage import destripe, destripe_lfp, detect_bad_channels, 
import spikeglx
import pandas as pd
import numpy as np
import scipy



def raw_data_qc(ap_cbin, lf_cbin, t_start, t_end):
    """
    Create a QC table with the following features for each channel:
    - AP_RAW_RMS
    - AP_BTR_RMS
    - AP_DST_RMS
    - AP_RES_RMS
    - AP_CHANNEL_LABELS
    - LF_RAW_RMS
    - LF_BTR_RMS
    - LF_DST_RMS
    - LF_RES_RMS
    - LF_CHANNEL_LABELS
    
    And the following columns overall:
    - AP_NUM_BAD_CHANNELS
    - LF_NUM_BAD_CHANNELS
    
    AP band data is filtered with a 300Hz high-pass butterworth for
    the BTR and DST features. LF band is filtered with a 2-200Hz band-pass
    butterworth.
    
    Each channel's feature is the median across snippets. 
    
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
        N=3, Wn=300/2*sr_ap.fs, btype="highpass", output="sos"
    )
    filter_lf = scipy.signal.butter(
        N=3, Wn=[2/2*sr_ap.fs, 200/2*sr_ap.fs], btype="bandpass", 
        output="sos"
    )
    
    for i, t0 in enumerate(t_start):
        # ap
        t1 = min(t_end[i], sr_ap.rl)
        sl = slice(int(t0*sr_ap.fs), int(t1*sr_ap.fs))
        raw = sr_ap[sl, :-sr_ap.nsync]
        
        
        
        
        
        
        
    
    
    
    
    