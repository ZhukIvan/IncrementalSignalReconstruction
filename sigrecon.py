import librosa
import librosa.filters
import numpy as np
import scipy as sp
from scipy import signal, fft, optimize


def load_wav(path):
  return librosa.core.load(path, sr=None)


def save_wav(wav, path, sample_rate):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  librosa.output.write_wav(path, wav.astype(np.int16), sr=sample_rate)


def snr(x, y):
  length = np.min((len(x), len(y))) - 1
  return 20*np.log10(np.linalg.norm(x[:length]) / np.linalg.norm(x[:length] - y[:length]))


def amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-6, x))


def db_to_amp(x):
  return np.power(10.0, x * 0.05)


def preemphasis(x, preemphasis):
  return signal.lfilter([1, -preemphasis], [1], x)


def normalize(S, min_level_db):
  return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def denormalize(S, min_level_db):
  return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


def inv_preemphasis(x, preemphasis):
  return signal.lfilter([1], [1, -preemphasis], x)


def objective_func(x0, path, x_fixed, m_i, n_fft):
  """Objective function, which would be minimized through reconstruction process

  Calculates magnitude of Fourier transform of the x signal through FFT and subtract m_i.
  Where x = [x_fixed, x0] when computing forward path, and x = [x0, x_fixed] for backward path.

  We have to separate x0 from x_fixed, because algorithm use x0 as the variable for optimization,
    and x_fixed as a static argument.

  Parameters
  ----------
  x0 : np.ndarray [shape=(hop_length, )]
    changeable part of the signal frame
  path : string
    direction to predict. 'forward' or 'backward'
  x_fixed : np.ndarray [shape=(win_length - hop_length, )]
    fixed part of the signal frame
  m_i : np.ndarray [shape=(n_fft / 2 + 1, )]
    i-th column of the spectrogram
  n_fft : int > 0 [scalar]
    FFT window size

  Returns
  -------
  y : np.ndarray [shape=(n_fft / 2 + 1, )]
    return difference between calculated specter magnitude of the x signal and m_i

  Notes
  -----
  This function can be modified in conjunction with stft calculation algorithm.
  """

  if path == 'forward':
    x = np.concatenate((x_fixed, x0))
  elif path == 'backward':
    x = np.concatenate((x0, x_fixed))
  return np.abs(fft(x, n_fft)[:n_fft / 2 + 1]) - m_i


def stft(y, n_fft=512, hop_length=64, win_length=256, dtype=np.complex64, window_type='boxcar'):
  """Short-time Fourier transform (STFT)

  Returns a complex-valued matrix D such that
    `np.abs(D[f, t])` is the magnitude of frequency bin `f` at frame `t`
    `np.angle(D[f, t])` is the phase of frequency bin `f` at frame `t`

  Parameters
  ----------
  y : np.ndarray [shape=(n,)], real-valued
    the input signal (audio time series)
  n_fft : int > 0 [scalar]
    Number of Fourier coefficients
  hop_length : int > 0 [scalar]
    The number of samples between successive frames
  win_length  : int <= n_fft [scalar]
    FFT window size
  window: string, float, or tuple
    The type of window to create. See 'scipy.signal.get_window' for more details
    At this point only 'boxcar' window is supported

  Notes
  ----
  Have to implement stft by my own, because librosa implementation is slightly unsuitable
  for this algorithm.
  """

  window = signal.get_window(window_type, win_length)

  # preallocate buffer for stft matrix
  stft_m = np.empty((int(n_fft / 2 + 1), (y.shape[0] - win_length) / hop_length + 1), dtype=dtype)

  # Go though number of frames
  for i in range(stft_m.shape[1]):
    # Get current frame from signal
    y_frame = y[i * hop_length : np.min((i * hop_length + win_length, y.shape[0] - 1))]
    # Calculate FFT
    stft_m[:, i] = fft(y_frame, n_fft)[:n_fft / 2 + 1]

  return stft_m


def lpc_coeffs(signal, order=16):
  """Compute the Linear Prediction Coefficients.

  Return the order LPC coefficients for the signal. c = lpc(x, k) will
  find the k coefficients of a k order linear filter:

    xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]

  Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.

  Parameters
  ----------
  signal: np.ndarray
    input signal
  order : int
    LPC order (the output will have order items)

  Notes
  ----
  This is just for reference, as it is using the direct inversion of the
  toeplitz matrix, which is really slow

  Basically algorithm should return order + 1 coefficients with 1 in the beginning, 
  but it's not necessary for prediction so algorithm return coefficients without 1.
  """

  p = order + 1
  r = np.zeros(p, signal.dtype)
  # Number of non zero values in autocorrelation one needs for p LPC
  # coefficients
  nx = np.min([p, signal.size])
  x = np.correlate(signal, signal, 'full')
  r[:nx] = x[signal.size - 1:signal.size + order]
  return np.dot(sp.linalg.pinv(sp.linalg.toeplitz(r[:-1])), -r[1:])


def lpc_predict(data, predict_len, lpc_order, path):
  """Predict sequence points, based on linear prediction coefficients.

  Return the predict_len number of predicted points based on data
  You can predict points after your sequence (forward path) or before(backward path)

  Parameters
  ----------
  data : np.ndarray
    input signal
  predict_len : int 
    number of points to predict
  lpc_order : int
    LPC order
  path : string
    direction to predict. 'forward' or 'backward'

  Returns
  -------
  predict : np.ndarray [shape=(predict_len, )]
    returns predicted sequence of points

  Notes
  ----
  If your predict_len greater than lpc_order, algorithm may behave very unstable
  """

  if(path == 'forward'):
    # we need to reverse coefficient to proper prediction
    lpc = -lpc_coeffs(data, lpc_order)

    # preallocate buffer with signal window in the beginning
    # for further placing points after signal window
    predict = np.concatenate((data[-lpc_order:], np.zeros(predict_len)))

    # compute dot product signal with coefficients and place it after signal window
    for i in np.arange(predict_len - 1):
      predict[i + lpc_order + 1] = np.dot(lpc, predict[i:i + lpc_order])

    # return only predicted part of signal
    return predict[lpc_order:]

  elif(path == 'backward'):
    lpc = -lpc_coeffs(data, lpc_order)

    # preallocate buffer with last signal window in the end
    # for further placing points before signal window
    predict = np.concatenate((np.zeros(predict_len), data[:lpc_order]))

    # compute dot product signal with coefficients and place it before signal window
    for i in np.arange(predict_len, 0, -1):
      predict[i - 1] = np.dot(lpc, predict[i:i + lpc_order])

    # return only predicted part of signal
    return predict[:predict_len]
  else:
    raise ValueError("path must be 'forward' or 'backward'")


def signal_reconstruction(S, n_fft=512, hop_length=64, win_length=256, err_tol=1e-02):
  """An Incremental Algorithm for Signal Reconstruction 
  from Short-Time Fourier Transform Magnitude

  Strongly recommended to use stft implementation from this module.
  All argument should be the same as in stft calculation.

  .. [1] An Incremental Algorithm for Signal Reconstruction from
          Short-Time Fourier Transform Magnitude
          J. Bouvrie and T. Ezzat, to appear at ICSLP 2006.

  Parameters
  ----------
  S : np.ndarray [shape=()]
  n_fft : int > 0 [scalar]
    Number of Fourier coefficients
  hop_length : int > 0 [scalar]
    The number of samples between successive frames
  win_length  : int <= n_fft [scalar]
    FFT window size
  err_tol: float
    Tolerance for termination by the change of the cost function. 
    See 'scipy.optimize.least_squares' for more information.

  Returns
  -------
  y : np.ndarray [shape=(n, )] 
    time domain signal reconstruction from spectrogram matrix S

  Notes
  ----
  All you want to know about original algorithm is listed in the paper.
  Only improvement I made is lpc prediction of initial guess,
  which work far better then original guesses.
  """

  # initialize
  x = np.zeros((S.T.shape[0], win_length))
  y = np.zeros(hop_length * (x.shape[0] - 1) + win_length)

  # Initial guess is positive, because signal is also have positive shift
  x[0] = np.random.uniform(0, 1, size=win_length)
  m = np.abs(S.T)

  # forward loop
  for i in np.arange(S.T.shape[0] - 1):
    # Predict initial guess points after x[i] frame
    x0 = lpc_predict(x[i], hop_length, hop_length, 'forward')

    # Get fixed part of the x[i] frame
    x_fixed = x[i][hop_length:]

    # Calculate precisely x0 as root finding problem
    sol = optimize.least_squares(objective_func, x0, args=('forward', x_fixed, m[i + 1], n_fft), 
                                 ftol=err_tol, method='lm')

    # Combine solution with fixed part into new x[i + 1] frame
    x[i + 1] = np.concatenate((x[i][hop_length:], sol.x))

    # Set the mean of the result to the value specified by the DC term of the Fourier magnitude
    x[i + 1] = x[i + 1] - np.mean(x[i + 1]) + m[i + 1][0] / win_length

  # backward loop
  for i in np.arange(S.T.shape[0] - 1, 0, -1):
    # Predict initial guess points before x[i] frame
    x0 = lpc_predict(x[i], hop_length, hop_length, 'backward')

    # Get fixed part of the x[i] frame
    x_fixed = x[i][:-hop_length]

    # Calculate precisely x0 as root finding problem
    sol = optimize.least_squares(objective_func, x0, args=('backward', x_fixed, m[i - 1], n_fft), 
                                 ftol=err_tol, method='lm')

    # Combine solution with fixed part into new x[i - 1] frame
    x[i - 1] = np.concatenate((sol.x, x[i][:-hop_length]))

    # Set the mean of the result to the value specified by the DC term of the Fourier magnitude
    x[i - 1] = x[i - 1] - np.mean(x[i - 1]) + m[i - 1][0] / win_length

  # Overlap adding
  for i in np.arange(x.shape[0]):
    pos = i * hop_length
    # Restore mean to 0 and normalize
    y[pos:pos + win_length] += (x[i] - np.mean(x[i])) / (win_length / hop_length)
  return y
