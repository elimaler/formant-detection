# formant-detection
Code for detecting formants from live audio.

This code is still in development.

The main function of this code is called by running `python3 main.py`. This code will start an audio session. Audio will be read in 1 second increaments. If vowel formants are detected in these increaments they will be printed.

## Tips
1. **Run this in a quiet envionrment.** This code does not filter out background-noise well. 
2. **Hold a singular vowel sound at a steady volume for a few seconds.** The code will detect the audio but only in a 1 second window. Holding for longer will give the code more of a chance to hear you.
3. **Allow for multiple windows of your vowel sound.** This code is still in development and so can be buggy. Try a few windows to get a good reading.
4. **Check the audio threshold.** If nothing is happening then your input may not be loud enough. Either adjust this or change the threshold in main.py

## Set-Up

To set up create a virtual environment using:

`python3 -m venv venv`

Start the environment with:

`source venv/bin/activate`

Once activated install requirements with:

`pip install -r requirements.txt`

Once installed start the program with:

`python3 main.py`
