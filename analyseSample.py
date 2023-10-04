import numpy as np
from pydub import AudioSegment
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def filter(frame, w):
    # Apply the median filter
    window_size = 2 * w + 1
    smoothed_frame = signal.medfilt(frame, kernel_size=window_size)

    # Smooth the result with a moving average
    smoothed_frame = signal.convolve(smoothed_frame, np.ones(150) / 150, mode='same')

    return smoothed_frame


def findFormantsFFT(audio_array, SAMPLE_RATE):
    
    fft_result = np.fft.fft(audio_array)

    # Calculate the frequency axis
    frequency_axis = np.fft.fftfreq(len(audio_array), d=1.0 / SAMPLE_RATE)[:len(audio_array) // 2]
    # Define the frequency range to keep (0-3500 Hz)
    frequency_min = 0
    frequency_max = 3500

    # Find the indices corresponding to the desired frequency range
    indices = np.where((frequency_axis >= frequency_min) & (frequency_axis <= frequency_max))

    # Extract the magnitude values for the selected frequency range
    filtered_magnitude = np.abs(fft_result[indices])
    filtered_magnitude = filter(filtered_magnitude, 100)
    # Create a filtered frequency axis
    filtered_frequency_axis = frequency_axis[indices]
    high_pass = np.array([1 / (1 + np.exp(-0.01 * (x - 100))) for x in filtered_frequency_axis])
    filtered_magnitude = filtered_magnitude * high_pass

    # Find the peaks in the magnitude data
    peaks, _ = signal.find_peaks(filtered_magnitude, height=0, distance=100)  # You may need to adjust the "height" and "distance" parameters
  

    # Sort the peaks by magnitude (highest first)
    sorted_peak_indices = np.argsort(filtered_magnitude[peaks])[::-1]
    sorted_peaks = peaks[sorted_peak_indices]

    # Extract the frequencies of the top three peaks
    top_three_peak_frequencies = filtered_frequency_axis[sorted_peaks[:3]]
    
    
    top_three_peak_frequencies.sort()
    r3_range = 2200
    i = 2
    if len(top_three_peak_frequencies)<3:
        return top_three_peak_frequencies
    while top_three_peak_frequencies[2] < r3_range:
        i += 1
        if sum([filtered_frequency_axis[i]>r3_range for i,x in enumerate(sorted_peaks[2:])]) == 0:
            return top_three_peak_frequencies[:2]
        else:
            top_three_peak_frequencies[2] = sorted_peaks[i]
    return top_three_peak_frequencies


    # # Plot the FFT output as a line plot
    # plt.figure(figsize=(10, 5))
    # plt.plot(filtered_frequency_axis, filtered_magnitude)  # Plot the magnitude of FFT
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title('FFT Output')
    # plt.grid(True)
    # plt.show()


def find_formants(audio_data, sample_rate, target_frame_rate=5):
    
    # Calculate the number of samples per frame to achieve the target frame rate
    samples_per_frame = int(sample_rate / target_frame_rate)
    
    # if the audio is tlass then 3 times the samples per frame return nothing
    if len(audio_data)<3*samples_per_frame:
         return [], []
    # Calculate a small overlap value
    overlap = int(samples_per_frame * 0.25)

    # Compute the spectrogram with the specified samples_per_frame and overlap
    frequencies, times, Sxx = signal.spectrogram(audio_data, fs=sample_rate, nperseg=samples_per_frame, noverlap=overlap, mode='magnitude')

    # define the frequency window
    start_frequency = 0
    end_frequency = 3500

    # Find the corresponding frequency indices in the spectrogram
    start_index = np.where(frequencies >= start_frequency)[0][0]
    end_index = np.where(frequencies >= end_frequency)[0][0]

    # trim the frequencies and SXX to window
    frequencies = frequencies[start_index:end_index]
    Sxx = Sxx[start_index:end_index]

    # apply a highpass filter to the audio
    high_pass = np.array([1 / (1 + np.exp(-0.06 * (i - 50))) for i, x in enumerate(Sxx.T[0])])
    Sxx = (Sxx * high_pass[:, np.newaxis])

    # apply the median and smoothing filters
    res = np.array([filter(x, 25) for x in Sxx.T])
    # make the whole spectogram the mean of this
    res = np.full_like(res, np.mean(res, axis=0)).T

    # Initial guess for the parameters (amplitudes, means, and std)
    initial_guess = [1000, 800, 600, 500, 1200, 3000, 100]
    # Fit the model to the data
    params, covariance = curve_fit(mixture_of_gaussians, frequencies, res.T[0], p0=initial_guess, maxfev=100000)
    # Extract the estimated parameters
    a1, mean1, a2, mean2, a3, mean3, std = params

    # create an array of the means
    means = [mean1, mean2, mean3]

    # if any means are below 100 or over 4000 then return nothing
    # this is erroneous data
    if sum([m<100 or m>4000 for m in means])>0:
        return [],None
    
    # sort the means
    means.sort()

    # if R1 and R2 are too close return nothing
    # this is erroneous data
    if means[1]-means[0]<50:
        return [],None

    Sxx = res
    data = (Sxx, frequencies, times, (start_frequency, end_frequency))
    return means, data

# Define the model as a mixture of three Gaussian distributions
def mixture_of_gaussians(x, a1, mean1, a2, mean2, a3, mean3, std):
    return (
        a1 * np.exp(-(x - mean1) ** 2 / (2 * std ** 2)) +
        a2 * np.exp(-(x - mean2) ** 2 / (2 * std ** 2)) +
        a3 * np.exp(-(x - mean3) ** 2 / (2 * std ** 2))
    )

def plot_spectogram(data):
    Sxx, frequencies, times, (start_frequency, end_frequency) = data
    # Define the custom frequency range (e.g., from 0 to 3500 Hz)
    # Create the spectrogram plot with the custom frequency range
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto', cmap='viridis')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram ({} Hz to {} Hz)'.format(start_frequency, end_frequency))
    plt.colorbar(label='Intensity (dB)')
    plt.show()


