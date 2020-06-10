import numpy as np
import nifty5 as ift
import phase_operator
import domainFlipper

class SymmetrizingOperator(ift.EndomorphicOperator):
    """Adds the field axes-wise in reverse order to itself.

    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        Domain of the operator.
    space : int
        Index of space in domain on which the operator shall act. Default is 0.
    """
    def __init__(self, domain, space=0):
        self._domain = ift.DomainTuple.make(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._space = ift.utilities.infer_space(self._domain, space)

    def apply(self, x, mode):
        self._check_input(x, mode)
        v = x.val.copy()
        for i in self._domain.axes[self._space]:
            lead = (slice(None),)*i
            v, loc = ift.dobj.ensure_not_distributed(v, (i,))
            loc[lead+(slice(None),)] += loc[lead+(slice(None, None, -1),)]
            loc /= 2
        return ift.Field(self.target, ift.dobj.ensure_default_distributed(v))


def get_likelihood(
    amp_dct,
    phase_dct,
    frequency_domain,
    large_frequency_domain,
    hardware_operator,
    filter_operator,
    fft_operator,
    data_traces,
    noise_levels
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
    domain_flipper = domainFlipper.DomainFlipper(zero_padder.domain, target = ift.RGSpace(small_sp.shape, harmonic=True))
    mag_S_h =  (domain_flipper @ zero_padder.adjoint @ correlated_field)
    mag_S_h = SymmetrizingOperator(mag_S_h.target) @ mag_S_h
    mag_S_h = realizer2.adjoint @ mag_S_h.exp()
    phi_S_h = phase_operator.SlopeSpectrumOperator(frequency_domain.get_default_codomain(), phase_dct['sm'], phase_dct['im'], phase_dct['sv'], phase_dct['iv'])
    phi_S_h = realizer2.adjoint @ phi_S_h


    efield_spec_operator = filter_operator @ (mag_S_h * (1.j*phi_S_h).exp())
    efield_trace_operator = realizer @ fft_operator.inverse @ efield_spec_operator

    channel_spec_operator = hardware_operator @ efield_spec_operator
    channel_trace_operator = realizer @ fft_operator.inverse @ channel_spec_operator

    likelihood = None
    for i_trace, data_trace in enumerate(data_traces):
        noise_operator = ift.ScalingOperator(noise_levels[i_trace]**2, frequency_domain.get_default_codomain())
        data_field = ift.Field(ift.DomainTuple.make(frequency_domain.get_default_codomain()), data_trace)
        if likelihood is None:
            likelihood = ift.GaussianEnergy(mean=data_field, inverse_covariance=noise_operator.inverse)(channel_trace_operator)
        else:
            likelihood += ift.GaussianEnergy(mean=data_field, inverse_covariance=noise_operator.inverse)(channel_trace_operator)

    return likelihood, efield_trace_operator, efield_spec_operator, channel_trace_operator, channel_spec_operator, A
