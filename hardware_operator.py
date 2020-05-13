import nifty5 as ift
import NuRadioReco.detector.RNO_G.analog_components
import NuRadioReco.detector.antennapattern
from NuRadioReco.utilities import units
import numpy as np
import scipy.signal


def get_hardware_operator(
        samples,
        sampling_rate,
        frequency_domain
    ):
    space_freqs = frequency_domain.get_k_length_array().val/samples*sampling_rate

    antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
    antenna_pattern = antenna_pattern_provider.load_antenna_pattern('greenland_vpol_InfFirn')
    antenna_res = antenna_pattern.get_antenna_response_vectorized(space_freqs, 90.*units.deg, 0, 0., 0.,  90.*units.deg, 0.)['theta']
    amp_response_func = NuRadioReco.detector.RNO_G.analog_components.load_amp_response('iglu')

    total_gain = np.abs(amp_response_func['gain'](space_freqs)) * np.abs(antenna_res)
    total_gain /= np.max(total_gain)
    amp_phase = np.unwrap(np.angle(amp_response_func['phase'](space_freqs)))
    total_phase = np.unwrap(np.angle(antenna_res))+amp_phase
    total_phase[len(total_phase)//2:] *= -1
    total_phase[len(total_phase)//2+1] = 0
    total_phase *= -1

    amp_field = ift.Field(ift.DomainTuple.make(frequency_domain), total_gain * np.exp(1.j*total_phase))
    amp_operator = ift.DiagonalOperator(amp_field)
    return amp_operator

def get_filter_operator(
        samples,
        sampling_rate,
        frequency_domain,
        passband = None
    ):
    if passband is None:
        return ift.ScalingOperator(1., frequency_domain)
    space_freqs = frequency_domain.get_k_length_array().val/samples*sampling_rate
    b, a = scipy.signal.butter(10, passband, 'bandpass', analog=True)
    w, op_h = scipy.signal.freqs(b, a, space_freqs)
    filter_field = ift.Field(ift.DomainTuple.make(frequency_domain), np.abs(op_h))
    filter_operator = ift.DiagonalOperator(filter_field, frequency_domain)
    return filter_operator
