import numpy as np
import nifty5 as ift
import phase_operator
import domainFlipper


def get_likelihood(
    amp_dct,
    phase_dct,
    frequency_domain,
    large_frequency_domain,
    hardware_operator,
    filter_operator,
    fft_operator,
    noise_operator,
    data_trace
    ):

    power_domain = ift.RGSpace(large_frequency_domain.get_default_codomain().shape[0], harmonic=True)
    power_space = ift.PowerSpace(power_domain)
    amp_dct['target'] = power_space
    A = ift.SLAmplitude(**amp_dct)
    correlated_field = ift.CorrelatedField(large_frequency_domain.get_default_codomain(), A)
    realizer = ift.Realizer(fft_operator.domain)
    realizer2 = ift.Realizer(fft_operator.target)
    large_sp = correlated_field.target
    small_sp = ift.RGSpace(large_sp.shape[0]//2, large_sp[0].distances)
    zero_padder = ift.FieldZeroPadder(small_sp, large_sp.shape, central=False)
    pd = zero_padder.adjoint @ correlated_field
    domain_flipper = domainFlipper.DomainFlipper(pd.target, target = ift.RGSpace(small_sp.shape, harmonic=True))
    mag_S_h =  (domain_flipper @ zero_padder.adjoint @ correlated_field)
    phi_S_h = phase_operator.SlopeSpectrumOperator(frequency_domain.get_default_codomain(), phase_dct['sm'], phase_dct['sv'], phase_dct['im'], phase_dct['iv'])
    efield_spec_operator = 5.e-2*(filter_operator @ realizer2.adjoint @ mag_S_h.exp()) * (1.j*realizer2.adjoint @ phi_S_h).exp()
    efield_trace_operator = realizer @ fft_operator.inverse @ efield_spec_operator

    channel_spec_operator = hardware_operator @ efield_spec_operator
    channel_trace_operator = realizer @ fft_operator.inverse @ channel_spec_operator

    data_field = ift.Field(ift.DomainTuple.make(frequency_domain.get_default_codomain()), data_trace)
    likelihood = ift.GaussianEnergy(mean=data_field, inverse_covariance=noise_operator.inverse)(channel_trace_operator)


    return likelihood, efield_trace_operator, efield_spec_operator, channel_trace_operator, channel_spec_operator
