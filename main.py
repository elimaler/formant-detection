import numpy as np
import pyaudio
import analyseSample
from collections import deque
from matplotlib import pyplot as plt

# activate this to make plot
# this uses a lot of performance and is prone to crash
PLOT = False

# Constants
SAMPLE_RATE = 44100  # Adjust as needed
CHUNK_SIZE = 4096*2   # Adjust as needed
MAX_PLOT_POINTS = 10  # Maximum number of formant triplets to display


def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')

    print("Available audio devices:")

    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        device_name = device_info.get('name')
        print(f"Device {i}: {device_name}")


# Initialize the audio stream
p = pyaudio.PyAudio()

# Get the input audio device's name
input_device_name = p.get_default_input_device_info().get('name')
print(f"Using input audio device: {input_device_name}")


stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)


if PLOT:
    # Create a deque to store the last 10 formant triplets
    formant_data = deque(maxlen=MAX_PLOT_POINTS)

    # Create a figure and axis for live plotting
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, MAX_PLOT_POINTS - 1)
    ax.set_ylim(0, 4000)  # Adjust the y-axis limit as needed

    # Create separate lines for each formant
    lines = [ax.plot([], [], marker='o')[0] for _ in range(3)]



try:
    print("STARTING... if n`othing happens either hold your vowel sound longer or adjust the treshold for audio.")
    print("Listening for vowel sounds...")
    pause_audio = False

    while True:
        # Read 1 second's worth of audio data
        audio_data = bytearray()
        # get the data for the second
        for _ in range(int(SAMPLE_RATE / CHUNK_SIZE)):
            audio_chunk = stream.read(CHUNK_SIZE)
            audio_data.extend(audio_chunk)
        

        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        # change this threshold for microphone.
        audio_threshold = 100
        # filter audio less than threshold
        audio_array = np.array([x for x in audio_array if np.abs(x)>audio_threshold])
        # only continue on audio longer than half a second
        if len(audio_array)>SAMPLE_RATE/2:
            # get formants
            formants, data = analyseSample.find_formants(audio_array, SAMPLE_RATE)
            # Display the detected harmonic frequencies
            if formants:
                # print found formants
                print("Detected formants:")
                print(f"\tR1: {formants[0]:.0f}Hz")
                print(f"\tR2: {formants[1]:.0f}Hz")
                print(f"\tR3: {formants[2]:.0f}Hz")
                if PLOT:
                    # Append the formant data to the deque
                    formant_data.append(formants)

                    # Update the live plot for each formant
                    x = np.arange(len(formant_data))
                    for i, line in enumerate(lines):
                        y = [f[i] for f in formant_data]
                        line.set_data(x, y)

                    ax.relim()
                    ax.autoscale_view()

                    plt.pause(0.01)  # Pause to update the plot
                # analyseSample.plot_spectogram(data)
        # Sleep for one second before reading the next chunk
        



except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()