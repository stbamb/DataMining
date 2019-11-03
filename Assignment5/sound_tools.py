# author        : Esteban
# course        : CS-691 Data Mining
# name          : sound_tools.py
# date          : 20191109
# python_version: 3.7
# notes         : Assignment5
# description   : Needed for main.py to run
# ==============================================================================

import librosa.feature
import numpy as np

import config


def getMFCCS(file_names):
    return [extractFeaturesFromAudio(file) for file in file_names]


# Taken from Dr. Khuri
# will extract 40 Mel-frequency cepstral coefficients
# these can be used as features to represent each wav file
# to learn more about MFCCS visit https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
def extractFeaturesFromAudio(file):
    data, sampling_rate = librosa.load(config.SRC_FOLDER + file)
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=config.NUMBER_OF_FEATURES_TO_EXTRACT)
    return np.mean(mfccs.T, axis=0).tolist()
