import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import fft, units, trace_utilities
import NuRadioMC.utilities.medium
import scipy.signal
import nifty5 as ift
import generate_data
import plotting
import hardware_operator
import likelihood


max_posterior = True
energy = 1.e18 * units.eV
medium = NuRadioMC.utilities.medium.get_ice_model('greenland_simple')
viewing_angle = 1. * units.deg
samples = 128
sampling_rate = 1. * units.GHz
model = 'ARZ2019'
shower_type = 'HAD'
noise_level = .2
passband = [120*units.MHz, 500*units.MHz]
amp_dct = {
    'n_pix': 32,  # 64 spectral bins

    # Spectral smoothness (affects Gaussian process part)
    'a': 1.8,  # relatively high variance of spectral curbvature
    'k0': 1.,  # quefrency mode below which cepstrum flattens

    # Power-law part of spectrum:
    'sm': -5.2,  # preferred power-law slope
    'sv': .1,  # low variance of power-law slope
    'im':  1.,  # y-intercept mean, in-/decrease for more/less contrast
    'iv': .4     # y-intercept variance
}
phase_dct = {
    'sm': 3.6,
    'sv': .1,
    'im': 0.,
    'iv': .5
}

efield_trace, noiseless_trace, voltage_trace, classic_efield_trace = generate_data.get_traces(
    energy = energy,
    viewing_angle = viewing_angle,
    samples = samples,
    sampling_rate = sampling_rate,
    shower_type = shower_type,
    medium =  medium,
    model = model,
    noise_level = noise_level,
    passband = passband
)
plotting.plot_data(
    efield_trace,
    noiseless_trace,
    voltage_trace,
    sampling_rate,
    'plots/data.png'
)

time_domain = ift.RGSpace(samples)
frequency_domain = ift.RGSpace(samples, harmonic=True)
large_frequency_domain = ift.RGSpace(samples*2, harmonic=True)

amp_operator = hardware_operator.get_hardware_operator(
        samples,
        sampling_rate,
        frequency_domain
)
filter_operator = hardware_operator.get_filter_operator(
    samples,
    sampling_rate,
    frequency_domain,
    passband = passband
)

fft_operator = ift.FFTOperator(frequency_domain.get_default_codomain())
noise_operator = ift.ScalingOperator(noise_level, frequency_domain.get_default_codomain())

likelihood, efield_trace_operator, efield_spec_operator, channel_trace_operator, channel_spec_operator = likelihood.get_likelihood(
    amp_dct,
    phase_dct,
    frequency_domain,
    large_frequency_domain,
    amp_operator,
    filter_operator,
    fft_operator,
    noise_operator,
    voltage_trace
    )
plotting.plot_priors(
    efield_spec_operator,
    efield_trace_operator,
    channel_spec_operator,
    channel_trace_operator,
    fft_operator,
    'plots/priors.png'
)


if max_posterior:
    ic_newton = ift.DeltaEnergyController(name='newton', iteration_limit=1000, tol_rel_deltaE=1e-9)
    minimizer = ift.NewtonCG(ic_newton)
    ic_sampling = ift.GradientNormController(iteration_limit=1000)
    H = ift.StandardHamiltonian(likelihood, ic_sampling)
    position = ift.from_random('normal', H.domain)
    for j in range(5):
        H = ift.StandardHamiltonian(likelihood, ic_sampling)
        H = ift.EnergyAdapter(position, H, want_metric=True)
        H, convergence = minimizer(H)
        plotting.plot_max_posterior(
            H,
            efield_trace,
            noiseless_trace,
            voltage_trace,
            classic_efield_trace,
            efield_trace_operator,
            channel_trace_operator,
            sampling_rate,
            'plots/max_posterior_reco_{}.png'.format(j)
        )
        median = H.position
        position = H.position
ic_newton = ift.DeltaEnergyController(name='newton', iteration_limit=100, tol_rel_deltaE=1e-7)
minimizer = ift.NewtonCG(ic_newton)
ic_sampling = ift.GradientNormController(iteration_limit=3000)
H = ift.StandardHamiltonian(likelihood, ic_sampling)
if not max_posterior:
    median = ift.MultiField.full(H.domain, 0.)
N_iterations = 30
N_samples = 30


for k in range(N_iterations):
    print('----------->>>   {}   <<<-----------'.format(k))
    KL = ift.MetricGaussianKL(median, H, N_samples)
    KL, convergence = minimizer(KL)
    median = KL.position
    plotting.plot_reco(
        KL,
        efield_trace,
        noiseless_trace,
        voltage_trace,
        classic_efield_trace,
        efield_trace_operator,
        channel_trace_operator,
        sampling_rate,
        'plots/reco_{}.png'.format(k)
    )
