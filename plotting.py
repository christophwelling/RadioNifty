import matplotlib.pyplot as plt
import numpy as np
import nifty5 as ift
from NuRadioReco.utilities import fft

def plot_data(
        efield_trace,
        noiseless_trace,
        voltage_trace,
        sampling_rate,
        noise_level,
        filename
    ):
    times = np.arange(len(voltage_trace))/sampling_rate
    freqs = np.fft.rfftfreq(len(voltage_trace), 1./sampling_rate)

    fig1 = plt.figure(figsize=(8,8))
    ax1_1 = fig1.add_subplot(221)
    ax1_2 = fig1.add_subplot(222)
    ax1_3 = fig1.add_subplot(223)
    ax1_4 = fig1.add_subplot(224)

    ax1_1.plot(times, efield_trace, c='C1')
    ax1_2.plot(freqs, np.abs(fft.time2freq(efield_trace, sampling_rate)), c='C1')
    ax1_3.plot(times, noiseless_trace, c='C1', alpha=1.)
    ax1_3.plot(times, voltage_trace, c='C0', alpha=.5)
    ax1_3.scatter(times, voltage_trace, c='C0')
    ax1_4.plot(freqs, np.abs(fft.time2freq(noiseless_trace, sampling_rate)), c='C1')
    ax1_4.plot(freqs, np.abs(fft.time2freq(voltage_trace, sampling_rate)), c='C0', alpha=.2)
    ax1_4.scatter(freqs, np.abs(fft.time2freq(voltage_trace, sampling_rate)), c='C0')

    ax1_1.grid()
    ax1_2.grid()
    ax1_3.grid()
    ax1_4.grid()
    snr_1 = .5 * (np.max(voltage_trace) - np.min(voltage_trace)) / noise_level
    snr_2 = .5 * (np.max(noiseless_trace) - np.min(noiseless_trace)) / noise_level
    ax1_3.text(.7, .95, 'SNR={:.1f}/{:.1f}'.format(snr_1, snr_2), transform=ax1_3.transAxes)
    fig1.tight_layout()
    fig1.savefig(filename)

def plot_priors(
        efield_spec_operator,
        efield_trace_operator,
        channel_spec_operator,
        channel_trace_operator,
        fft_operator,
        power_operator,
        filename
    ):

    fig1 = plt.figure(figsize=(8,12))
    ax1_1 = fig1.add_subplot(421)
    ax1_2 = fig1.add_subplot(422)
    ax1_3 = fig1.add_subplot(4,2,3)
    ax1_4 = fig1.add_subplot(425)
    ax1_5 = fig1.add_subplot(426)
    ax1_6 = fig1.add_subplot(4,2,(7,8))
    ax1_7 = fig1.add_subplot(424)
    real_efield_spec_operator = fft_operator @ efield_trace_operator
    real_channel_spec_operator =  fft_operator @ channel_trace_operator
    freq_space = efield_spec_operator.target[0]
    for i in range(10):
        x = ift.from_random('normal', efield_spec_operator.domain)
        efield_spec_sample = efield_spec_operator.force(x)
        real_efield_spec_sample = real_efield_spec_operator.force(x)
        efield_trace_sample = efield_trace_operator.force(x)
        channel_spec_sample = channel_spec_operator.force(x)
        real_channel_spec_sample = real_channel_spec_operator.force(x)
        channel_trace_sample = channel_trace_operator.force(x)
        ax1_1.plot(freq_space.get_k_length_array().val, np.abs(efield_spec_sample.val)/np.max(np.abs(efield_spec_sample.val)), c='C{}'.format(i), linestyle=':')
        ax1_1.plot(freq_space.get_k_length_array().val, np.abs(real_efield_spec_sample.val)/np.max(np.abs(real_efield_spec_sample.val)), c='C{}'.format(i))

        ax1_2.plot(freq_space.get_k_length_array().val, np.abs(efield_spec_sample.val), c='C{}'.format(i), linestyle=':')
        ax1_2.plot(freq_space.get_k_length_array().val, np.abs(real_efield_spec_sample.val), c='C{}'.format(i))

        ax1_3.plot(efield_trace_sample.val/np.max(np.abs(efield_trace_sample.val)))

        ax1_4.plot(freq_space.get_k_length_array().val, np.abs(channel_spec_sample.val)/np.max(np.abs(channel_spec_sample.val)), c='C{}'.format(i), linestyle=':')
        ax1_4.plot(freq_space.get_k_length_array().val, np.abs(real_channel_spec_sample.val)/np.max(np.abs(real_channel_spec_sample.val)), c='C{}'.format(i))
        ax1_5.plot(freq_space.get_k_length_array().val, np.abs(channel_spec_sample.val), c='C{}'.format(i), linestyle=':')
        ax1_5.plot(freq_space.get_k_length_array().val, np.abs(real_channel_spec_sample.val), c='C{}'.format(i))

        ax1_6.plot(channel_trace_sample.val/np.max(np.abs(channel_trace_sample.val)), c='C{}'.format(i))

        power_spectrum = power_operator.force(x)
        ax1_7.plot(power_spectrum.domain[0].k_lengths, power_spectrum.val, c='C{}'.format(i))

    ax1_1.grid()
    ax1_2.grid()
    ax1_2.set_yscale('log')
    ax1_2.set_ylim([1.e-4, None])
    ax1_3.grid()
    ax1_3.set_xlim([40,100])
    ax1_4.grid()
    ax1_5.grid()
    ax1_5.set_yscale('log')
    ax1_5.set_ylim([1.e-5, None])
    ax1_6.grid()
    ax1_7.set_xscale('log')
    ax1_7.set_yscale('log')
    ax1_7.grid()
    fig1.tight_layout()
    fig1.savefig(filename)

def plot_reco(
    KL,
    efield_trace,
    noiseless_trace,
    voltage_trace,
    classic_efield_trace,
    efield_trace_operator,
    channel_trace_operator,
    sampling_rate,
    noise_rms,
    filename
    ):
    plt.close('all')
    times = np.arange(len(voltage_trace))/sampling_rate
    freqs = np.fft.rfftfreq(len(voltage_trace), 1./sampling_rate)

    fig1 = plt.figure(figsize=(12,10))
    ax1_1 = fig1.add_subplot(4,3,(1,4))
    ax1_2 = fig1.add_subplot(4,3,(2,5))
    ax1_3 = fig1.add_subplot(4,3,(7,10))
    ax1_4 = fig1.add_subplot(4,3,(8,11))
    ax1_5 = fig1.add_subplot(4,3,3)
    ax1_6 = fig1.add_subplot(4,3,6)
    ax1_7 = fig1.add_subplot(4,3,9)
    ax1_8 = fig1.add_subplot(4,3,12)

    trace_stat_calculator = ift.StatCalculator()
    efield_stat_calculator = ift.StatCalculator()
    amp_trace_stat_calculator = ift.StatCalculator()
    amp_efield_stat_calculator = ift.StatCalculator()
    median = KL.position
    ax1_1.plot(times, efield_trace, c='C1', label='MC truth', linewidth=5)
    ax1_2.plot(freqs, np.abs(fft.time2freq(efield_trace, sampling_rate)), c='C1', linewidth=5)
    for sample in KL.samples:
        channel_sample_trace = channel_trace_operator.force(median + sample).val
        trace_stat_calculator.add(channel_sample_trace)
        amp_trace = np.abs(fft.time2freq(channel_sample_trace, sampling_rate))
        amp_trace_stat_calculator.add(amp_trace)
        #ax1_3.plot(times, channel_sample_trace, c='k', alpha=.1)
        ax1_4.plot(freqs, amp_trace, c='k', alpha=.1)
        efield_sample_trace = efield_trace_operator.force(median + sample).val
        efield_stat_calculator.add(efield_sample_trace)
        amp_efield = np.abs(fft.time2freq(efield_sample_trace, sampling_rate))
        amp_efield_stat_calculator.add(amp_efield)
        #ax1_1.plot(times, efield_sample_trace, c='k', alpha=.1)
        ax1_2.plot(freqs, amp_efield, c='k', alpha=.1)

    ax1_3.plot(times, noiseless_trace, c='C1', label='MC truth', linewidth=5, zorder=0)
    ax1_3.plot(times, voltage_trace, c='C0', alpha=.2, zorder=10)
    ax1_3.scatter(times, voltage_trace, c='C0', label='data', zorder=10)

    ax1_4.plot(freqs, np.abs(fft.time2freq(noiseless_trace, sampling_rate)), c='C1', linewidth=5, zorder=0)
    ax1_4.plot(freqs, np.abs(fft.time2freq(voltage_trace, sampling_rate)), c='C0', alpha=.2, zorder=10)
    ax1_4.scatter(freqs, np.abs(fft.time2freq(voltage_trace, sampling_rate)), c='C0', zorder=10)


    ax1_1.plot(times, efield_stat_calculator.mean, c='C2', label='max. posterior')
    ax1_1.plot(times, classic_efield_trace, c='C0', alpha=.5, label='classic reco')
    ax1_1.scatter(times, classic_efield_trace, c='C0', alpha=1.)
    ax1_2.plot(freqs, amp_efield_stat_calculator.mean, c='C2')
    ax1_2.plot(freqs, np.abs(fft.time2freq(classic_efield_trace, sampling_rate)), c='C0', alpha=.5)
    ax1_2.scatter(freqs, np.abs(fft.time2freq(classic_efield_trace, sampling_rate)), c='C0', alpha=1.)
    ax1_3.plot(times, trace_stat_calculator.mean, c='C2', label='max. posterior', zorder=5)
    ax1_4.plot(freqs, amp_trace_stat_calculator.mean, c='C2', zorder=5)

    ax1_1.set_ylim([1.5*np.min(efield_trace), 1.5*np.max(efield_trace)])
    ax1_2.set_ylim([0, 1.5*np.max(np.abs(fft.time2freq(efield_trace, sampling_rate)))])

    ax1_5.plot(times, efield_stat_calculator.mean - efield_trace, c='C2')
    ax1_5.plot(times, classic_efield_trace - efield_trace, c='C0', alpha=.5)
    ax1_5.scatter(times, classic_efield_trace - efield_trace, c='C0', alpha=1.)
    ax1_6.plot(freqs, np.abs(amp_efield_stat_calculator.mean) - np.abs(fft.time2freq(efield_trace, sampling_rate)), c='C2')
    ax1_6.plot(freqs, np.abs(fft.time2freq(classic_efield_trace, sampling_rate)) - np.abs(fft.time2freq(efield_trace, sampling_rate)), c='C0', alpha=.5)
    ax1_6.scatter(freqs, np.abs(fft.time2freq(classic_efield_trace, sampling_rate)) - np.abs(fft.time2freq(efield_trace, sampling_rate)), c='C0', alpha=1.)

    ax1_7.plot(times, trace_stat_calculator.mean - noiseless_trace, c='C2')
    ax1_7.plot(times, voltage_trace - noiseless_trace, c='C0', alpha=.5)
    ax1_7.scatter(times, voltage_trace - noiseless_trace, c='C0', alpha=1.)
    ax1_8.plot(freqs, np.abs(amp_trace_stat_calculator.mean) - np.abs(fft.time2freq(noiseless_trace, sampling_rate)), c='C2')
    ax1_8.plot(freqs, np.abs(fft.time2freq(voltage_trace, sampling_rate)) - np.abs(fft.time2freq(noiseless_trace, sampling_rate)), c='C0', alpha=.5)
    ax1_8.scatter(freqs, np.abs(fft.time2freq(voltage_trace, sampling_rate)) - np.abs(fft.time2freq(noiseless_trace, sampling_rate)), c='C0', alpha=1.)

    snr_1 = .5 * (np.max(voltage_trace) - np.min(voltage_trace)) / noise_rms
    snr_2 = .5 * (np.max(noiseless_trace) - np.min(noiseless_trace)) / noise_rms

    ax1_1.grid()
    ax1_1.legend()
    ax1_2.grid()
    ax1_3.grid()
    ax1_3.legend()
    ax1_3.text(.55, .05, 'SNR={:.2f}/{:.2f}'.format(snr_1, snr_2), transform=ax1_3.transAxes, bbox=dict(facecolor='white', alpha=.5, ec='white'), fontsize=12)
    ax1_4.grid()
    ax1_5.grid()
    ax1_6.grid()
    ax1_7.grid()
    ax1_8.grid()
    ax1_1.set_xlabel('t [ns]')
    ax1_1.set_ylabel('E [a.u.]')
    ax1_2.set_xlabel('f [GHz]')
    ax1_2.set_ylabel('E [a.u.]')
    ax1_3.set_xlabel('t [ns]')
    ax1_3.set_ylabel('U [a.u.]')
    ax1_4.set_xlabel('f [GHz]')
    ax1_4.set_ylabel('U [a.u.]')
    ax1_5.set_xlabel('t [ns]')
    ax1_5.set_ylabel('E residuals [a.u.]')
    ax1_6.set_xlabel('f [GHz]')
    ax1_6.set_ylabel('E residuals [a.u.]')
    ax1_7.set_xlabel('t [ns]')
    ax1_7.set_ylabel('U residuals [a.u.]')
    ax1_8.set_xlabel('f [GHz]')
    ax1_8.set_ylabel('U residuals [a.u.]')

    fig1.tight_layout()
    fig1.savefig(filename)
def plot_max_posterior(
    KL,
    efield_trace,
    noiseless_trace,
    voltage_trace,
    classic_efield_trace,
    efield_trace_operator,
    channel_trace_operator,
    sampling_rate,
    filename
    ):

    times = np.arange(len(voltage_trace))/sampling_rate
    freqs = np.fft.rfftfreq(len(voltage_trace), 1./sampling_rate)

    fig1 = plt.figure(figsize=(10,12))
    ax1_1 = fig1.add_subplot(221)
    ax1_2 = fig1.add_subplot(222)
    ax1_3 = fig1.add_subplot(223)
    ax1_4 = fig1.add_subplot(224)

    ax1_1.plot(times, efield_trace, c='C1', label='MC truth')
    ax1_2.plot(freqs, np.abs(fft.time2freq(efield_trace, sampling_rate)), c='C1')

    ax1_3.plot(times, noiseless_trace, c='C1', label='MC truth')
    ax1_3.plot(times, voltage_trace, c='C0', alpha=.2)
    ax1_3.scatter(times, voltage_trace, c='C0', label='data')

    ax1_4.plot(freqs, np.abs(fft.time2freq(noiseless_trace, sampling_rate)), c='C1')
    ax1_4.plot(freqs, np.abs(fft.time2freq(voltage_trace, sampling_rate)), c='C0', alpha=.2)
    ax1_4.scatter(freqs, np.abs(fft.time2freq(voltage_trace, sampling_rate)), c='C0')


    median = KL.position
    channel_rec_trace = channel_trace_operator.force(median).val
    efield_rec_trace = efield_trace_operator.force(median).val
    ax1_1.plot(times, efield_rec_trace, c='C2', label='max. posterior')
    ax1_1.plot(times, classic_efield_trace, c='C0', alpha=.5, label='classic reco')
    ax1_2.plot(freqs, np.abs(fft.time2freq(efield_rec_trace, sampling_rate)), c='C2')
    ax1_2.plot(freqs, np.abs(fft.time2freq(classic_efield_trace, sampling_rate)), c='C0', alpha=.5)
    ax1_3.plot(times, channel_rec_trace, c='C2', label='max. posterior')
    ax1_4.plot(freqs, np.abs(fft.time2freq(channel_rec_trace, sampling_rate)), c='C2')
    ax1_1.set_ylim([1.5*np.min(efield_trace), 1.5*np.max(efield_trace)])
    ax1_2.set_ylim([0, 1.5*np.max(np.abs(fft.time2freq(efield_trace, sampling_rate)))])
    ax1_1.grid()
    ax1_1.legend()
    ax1_2.grid()
    ax1_3.grid()
    ax1_3.legend()
    ax1_4.grid()

    fig1.tight_layout()
    fig1.savefig(filename)

def plot_energies(energies, filename):
    plt.close('all')
    fig1 = plt.figure()
    ax1_1 = fig1.add_subplot(111)
    ax1_1.scatter(np.arange(len(energies)), energies)
    ax1_1.axvline(np.argmin(energies), c='k', alpha=.5)
    ax1_1.grid()
    #ax1_1.set_yscale('log')
    fig1.tight_layout()
    fig1.savefig(filename)
