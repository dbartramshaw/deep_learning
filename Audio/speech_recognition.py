import os
from pocketsphinx import LiveSpeech, get_model_path

model_path = get_model_path()

speech = LiveSpeech(
    verbose=False,
    sampling_rate=16000,
    buffer_size=2048,
    no_search=False,
    full_utt=False,
    hmm=os.path.join(model_path, 'en-us'),
    lm=os.path.join(model_path, 'en-us.lm.bin'),
    dic=os.path.join(model_path, 'cmudict-en-us.dict'))

for phrase in speech:
    print(phrase)


from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

import os
import pyaudio
import wave
import audioop
from collections import deque
import time
import math

"""
Written by Sophie Li, 2016
http://blog.justsophie.com/python-speech-to-text-with-pocketsphinx/
"""

class SpeechDetector:
    def __init__(self):
        # Microphone stream config.
        self.CHUNK = 1024  # CHUNKS of bytes to read each time from mic
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        self.SILENCE_LIMIT = 1  # Silence limit in seconds. The max ammount of seconds where
                           # only silence is recorded. When this time passes the
                           # recording finishes and the file is decoded

        self.PREV_AUDIO = 0.5  # Previous audio (in seconds) to prepend. When noise
                          # is detected, how much of previously recorded audio is
                          # prepended. This helps to prevent chopping the beginning
                          # of the phrase.

        self.THRESHOLD = 4500
        self.num_phrases = -1

        # These will need to be modified according to where the pocketsphinx folder is
        MODELDIR = "../../tools/pocketsphinx/model"
        DATADIR = "../../tools/pocketsphinx/test/data"

        # Create a decoder with certain model
        config = Decoder.default_config()
        config.set_string('-hmm', os.path.join(MODELDIR, 'en-us/en-us'))
        config.set_string('-lm', os.path.join(MODELDIR, 'en-us/en-us.lm.bin'))
        config.set_string('-dict', os.path.join(MODELDIR, 'en-us/cmudict-en-us.dict'))

        # Creaders decoder object for streaming data.
        self.decoder = Decoder(config)

    def setup_mic(self, num_samples=50):
        """ Gets average audio intensity of your mic sound. You can use it to get
            average intensities while you're talking and/or silent. The average
            is the avg of the .2 of the largest intensities recorded.
        """
        print "Getting intensity values from mic."
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        values = [math.sqrt(abs(audioop.avg(stream.read(self.CHUNK), 4)))
                  for x in range(num_samples)]
        values = sorted(values, reverse=True)
        r = sum(values[:int(num_samples * 0.2)]) / int(num_samples * 0.2)
        print " Finished "
        print " Average audio intensity is ", r
        stream.close()
        p.terminate()

        if r < 3000:
            self.THRESHOLD = 3500
        else:
            self.THRESHOLD = r + 100

    def save_speech(self, data, p):
        """
        Saves mic data to temporary WAV file. Returns filename of saved
        file
        """
        filename = 'output_'+str(int(time.time()))
        # writes data to WAV file
        data = ''.join(data)
        wf = wave.open(filename + '.wav', 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)  # TODO make this value a function parameter?
        wf.writeframes(data)
        wf.close()
        return filename + '.wav'

    def decode_phrase(self, wav_file):
        self.decoder.start_utt()
        stream = open(wav_file, "rb")
        while True:
          buf = stream.read(1024)
          if buf:
            self.decoder.process_raw(buf, False, False)
          else:
            break
        self.decoder.end_utt()
        words = []
        [words.append(seg.word) for seg in self.decoder.seg()]
        return words

    def run(self):
        """
        Listens to Microphone, extracts phrases from it and calls pocketsphinx
        to decode the sound
        """
        self.setup_mic()

        #Open stream
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        print "* Mic set up and listening. "

        audio2send = []
        cur_data = ''  # current chunk of audio data
        rel = self.RATE/self.CHUNK
        slid_win = deque(maxlen=self.SILENCE_LIMIT * rel)
        #Prepend audio from 0.5 seconds before noise was detected
        prev_audio = deque(maxlen=self.PREV_AUDIO * rel)
        started = False

        while True:
            cur_data = stream.read(self.CHUNK)
            slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))

            if sum([x > self.THRESHOLD for x in slid_win]) > 0:
                if started == False:
                    print "Starting recording of phrase"
                    started = True
                audio2send.append(cur_data)

            elif started:
                print "Finished recording, decoding phrase"
                filename = self.save_speech(list(prev_audio) + audio2send, p)
                r = self.decode_phrase(filename)
                print "DETECTED: ", r

                # Removes temp audio file
                os.remove(filename)
                # Reset all
                started = False
                slid_win = deque(maxlen=self.SILENCE_LIMIT * rel)
                prev_audio = deque(maxlen=0.5 * rel)
                audio2send = []
                print "Listening ..."

            else:
                prev_audio.append(cur_data)

        print "* Done listening"
        stream.close()
        p.terminate()

if __name__ == "__main__":
    sd = SpeechDetector()
    sd.run()

0.006*4*60/2

##################################################################
#
##################################################################

# Convert Files
ffmpeg -i "/Users/bartramshawd/Downloads/Aaliki Agathangelou.m4a" -f flac "/Users/bartramshawd/Downloads/Aaliki_Agathangelou.flac"



#!/usr/bin/env python3

import speech_recognition as sr

# obtain path to "english.wav" in the same folder as this script
from os import path
AUDIO_FILE = "/Users/bartramshawd/Downloads/audio-file.flac"
AUDIO_FILE = "/Users/bartramshawd/Downloads/Aaliki_Agathangelou.flac"
# AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "english.wav")
# AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "french.aiff")
# AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "chinese.flac")

# use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # read the entire audio file



# recognize speech using Sphinx
try:
    print("Sphinx thinks you said " + r.recognize_sphinx(audio))
except sr.UnknownValueError:
    print("Sphinx could not understand audio")
except sr.RequestError as e:
    print("Sphinx error; {0}".format(e))
    text_output = e
    test_output_2 = "Sphinx error; {0}".format(e)


print text_output


# recognize speech using Google Speech Recognition
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    print("Google Speech Recognition thinks you said " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

# recognize speech using Google Cloud Speech
GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""INSERT THE CONTENTS OF THE GOOGLE CLOUD SPEECH JSON CREDENTIALS FILE HERE"""
try:
    print("Google Cloud Speech thinks you said " + r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS))
except sr.UnknownValueError:
    print("Google Cloud Speech could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Cloud Speech service; {0}".format(e))

# recognize speech using Wit.ai
WIT_AI_KEY = "INSERT WIT.AI API KEY HERE"  # Wit.ai keys are 32-character uppercase alphanumeric strings
try:
    print("Wit.ai thinks you said " + r.recognize_wit(audio, key=WIT_AI_KEY))
except sr.UnknownValueError:
    print("Wit.ai could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Wit.ai service; {0}".format(e))

# recognize speech using Microsoft Bing Voice Recognition
BING_KEY = "INSERT BING API KEY HERE"  # Microsoft Bing Voice Recognition API keys 32-character lowercase hexadecimal strings
try:
    print("Microsoft Bing Voice Recognition thinks you said " + r.recognize_bing(audio, key=BING_KEY))
except sr.UnknownValueError:
    print("Microsoft Bing Voice Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Microsoft Bing Voice Recognition service; {0}".format(e))

# recognize speech using Houndify
HOUNDIFY_CLIENT_ID = "INSERT HOUNDIFY CLIENT ID HERE"  # Houndify client IDs are Base64-encoded strings
HOUNDIFY_CLIENT_KEY = "INSERT HOUNDIFY CLIENT KEY HERE"  # Houndify client keys are Base64-encoded strings
try:
    print("Houndify thinks you said " + r.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY))
except sr.UnknownValueError:
    print("Houndify could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Houndify service; {0}".format(e))

# recognize speech using IBM Speech to Text
IBM_USERNAME = "INSERT IBM SPEECH TO TEXT USERNAME HERE"  # IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
IBM_PASSWORD = "INSERT IBM SPEECH TO TEXT PASSWORD HERE"  # IBM Speech to Text passwords are mixed-case alphanumeric strings
try:
    print("IBM Speech to Text thinks you said " + r.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD))
except sr.UnknownValueError:
    print("IBM Speech to Text could not understand audio")
except sr.RequestError as e:
    print("Could not request results from IBM Speech to Text service; {0}".format(e))
