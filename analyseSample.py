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
    smoothed_frame = signal.convolve(smoothed_frame, np.ones(20) / 20, mode='same')

    return smoothed_frame


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








# # Load the MP3 audio file using pydub
# audio = AudioSegment.from_mp3('u.mp3')  # Replace with your .mp3 file

# # Convert to numpy array
# audio_data = np.array(audio.get_array_of_samples())

# # Sample rate of the audio data (adjust if needed)
# sample_rate = audio.frame_rate


# (mean1, mean2, mean3), (Sxx, frequencies, times, (start_frequency, end_frequency)) = find_formants(audio_data, sample_rate)
# # plot_spectogram(Sxx, frequencies, times, start_frequency, end_frequency)
