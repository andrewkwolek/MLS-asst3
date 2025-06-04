# VGGish parameters for audio processing

# Sample rate for audio input
SAMPLE_RATE = 16000

# Number of samples in an input frame
FRAME_SIZE = 0.96  # seconds

# Number of samples between consecutive frames
HOP_SIZE = 0.96  # seconds

# Number of mel frequency bands for the mel spectrogram
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
NUM_MEL_BINS = 64

# Number of frames in the input to the model
NUM_FRAMES = 96

# Dimensions of the input to the model
EXPECTED_SHAPE = (NUM_FRAMES, NUM_MEL_BINS, 1)

# Quantization parameters for converting between floating point and integer
QUANTIZE_MIN_VAL = -2.0
QUANTIZE_MAX_VAL = +2.0

# VGGish expects log mel spectrograms that are normalized to [0, 1]
# The original VGGish model was trained on AudioSet which used these specific
# preprocessing parameters.
