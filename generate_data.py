import NuRadioReco.detector.RNO_G.analog_components
import NuRadioReco.detector.antennapattern
import NuRadioMC.SignalGen.askaryan
import scipy.signal
import numpy as np
from NuRadioReco.utilities import fft, units

def get_traces(
        energy,
        viewing_angle,
        samples,
        sampling_rate,
        shower_type,
        medium,
        model,
        noise_level,
        passband=None
    ):
    ior = medium.get_index_of_refraction([0,0,-450])
    cherenkov_angle = np.arccos(1./ior)

    efield_trace = NuRadioMC.SignalGen.askaryan.get_time_trace(
        energy = energy,
        theta = viewing_angle + cherenkov_angle,
        N = samples,
        dt = 1./sampling_rate,
        shower_type = shower_type,
        n_index = ior,
        R = 1000.,
        model = model
    )

    antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
    antenna_pattern = antenna_pattern_provider.load_antenna_pattern('greenland_vpol_InfFirn')
    freqs = np.fft.rfftfreq(samples, 1./sampling_rate)
    times = np.arange(samples) / sampling_rate

    if passband is None:
        h = np.ones_like(freqs)
    else:
        b, a = scipy.signal.butter(10,passband, 'bandpass', analog=True)
        w, h = scipy.signal.freqs(b, a, freqs)
    efield_trace = fft.freq2time(fft.time2freq(efield_trace, sampling_rate)*np.abs(h),sampling_rate)
    efield_trace = np.roll(efield_trace, 10)
    efield_trace /= np.max(np.abs(efield_trace))
    efield_spectrum = fft.time2freq(efield_trace, sampling_rate)
    antenna_response = antenna_pattern.get_antenna_response_vectorized(freqs, 90.*units.deg, 0, 0., 0.,  90.*units.deg, 0.)['theta']
    amp_response_func = NuRadioReco.detector.RNO_G.analog_components.load_amp_response('iglu')
    amp_gain = amp_response_func['gain'](freqs)
    amp_phase = amp_response_func['phase'](freqs)

    noiseless_spectrum = antenna_response * efield_spectrum * amp_gain * amp_phase / np.max(np.abs(antenna_response*amp_gain))
    noiseless_trace = fft.freq2time(noiseless_spectrum, sampling_rate)
    trace_max = np.max(np.abs(noiseless_trace))
    efield_trace /= trace_max
    noiseless_trace /= trace_max

    noise = np.random.normal(0, noise_level, samples)
    voltage_trace = noiseless_trace + noise
    classic_efield_spec = fft.time2freq(voltage_trace, sampling_rate) / (antenna_response * amp_gain * amp_phase / np.max(np.abs(antenna_response*amp_gain)))
    classic_efield_spec[~np.isfinite(classic_efield_spec)] = 0
    if passband is not None:
        classic_efield_spec *= np.abs(h)

    classic_efield_trace = fft.freq2time(classic_efield_spec, sampling_rate)
    trace_max = np.max(np.abs(voltage_trace))
    efield_trace /= trace_max
    noiseless_trace /= trace_max
    voltage_trace /= trace_max
    classic_efield_trace /= trace_max

    return efield_trace, noiseless_trace, voltage_trace, classic_efield_trace
