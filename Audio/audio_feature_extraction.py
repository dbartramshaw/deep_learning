#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Audio Extraction

    Based on https://musicinformationretrieval.com/
"""

#http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib
import essentia, essentia.standard as ess
plt.rcParams['figure.figsize'] = (14,4)


# ##################
# # Load from URL
# ##################
# url = 'http://page.subpage.com/file_name.wav'
# from urllib.request import urlretrieve
# urlretrieve(url, filename='simple_loop.wav')

####################
# Load from file
####################
x, fs = librosa.load('Audio/sample_audio.WAV')
librosa.display.waveplot(x, sr=fs)
