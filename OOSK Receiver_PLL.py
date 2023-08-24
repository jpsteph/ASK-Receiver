

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1:] / n

#numpy type can either be "np.complex128" or "np.complex64"
def get_samples_from_file(file_name, bin = True, numpytype = str):
    if bin == True:
        if numpytype == "complex128":
            return np.fromfile(file_name +".bin", dtype=np.complex128)
        elif numpytype == "complex64":
            return np.fromfile(file_name +".bin", dtype=np.complex64)
    else:
        if numpytype == "complex128":
            return np.fromfile(file_name, dtype=np.complex128)
        elif numpytype == "complex64":
            return np.fromfile(file_name, dtype=np.complex64)

def spectogram(IQ_sig, bins, Fs, graph=None, n_point_fft=None):
    """
    Compute the spectrogram of a signal.

    Parameters:
        sig (array-like): Input IQ signal data.
        bins (int): Number of FFT bins for each segment.
        Fs (float): Sampling frequency of the signal.
        graph (optional): If not None, display the spectrogram plot.
        specto_resolution (optional): Number of FFT points for spectral resolution.

    Returns:
        array-like: Spectrogram matrix.

    """

    num_rows = len(IQ_sig) // bins

    if n_point_fft is not None:   #creating spectrogram matrix to store resulting ffts (column size is dictated by fft size)
        spectrogram = np.zeros((num_rows, n_point_fft))
    else:
        spectrogram = np.zeros((num_rows, bins))

    for i in range(num_rows): 
        #getting section of samples to do fft
        start = i * bins
        end = (i + 1) * bins
        segment = IQ_sig[start:end]

        if n_point_fft is not None:
            spectrum = np.fft.fftshift(np.fft.fft(segment, n_point_fft))    #doing n-point fft dicated by n_point_fft
        else:
            spectrum = np.fft.fftshift(np.fft.fft(segment)) #doing fft of resolution based on size of sample segment (larger segments will result in a higher resolution fft)

        spectrogram[i, :] = 10 * np.log10(np.square(np.abs(spectrum)))    #getting magnitude of fft and converting to dbm and storing fft in row in spectrogram

    if graph is not None:
                #x labels                           #y labels
        extent = [(-Fs / 2) / 1e6, (Fs / 2) / 1e6, len(IQ_sig) / Fs, 0] #spectrogram goes top to bottom
        plt.imshow(spectrogram, aspect='auto', extent=extent)
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [s]")
        plt.show()

    return spectrogram

def find_clock_frequency(spectrum):
    maxima = scipy.signal.argrelextrema(spectrum, np.greater_equal)[0]
    while maxima[0] < 2:
        maxima = maxima[1:]
    if maxima.any():
        threshold = max(spectrum[2:-1])*0.8
        indices_above_threshold = np.argwhere(spectrum[maxima] > threshold)
        return maxima[indices_above_threshold[0]]
    else:
        return 0

def simple_psd(buff, NFFT, Fs, frf):
    plt.psd(buff, #data buffer
    NFFT, #FFT Size
    Fs, #Sampling Rate
    frf, #Center Frequency (Hz)
    scale_by_freq = True
    ) 
    plt.show()

def find_bit_sequence(binary_list, sequence):
    sequence_length = len(sequence)
    for i in range(len(binary_list) - sequence_length + 1):
        if binary_list[i:i + sequence_length] == sequence:
            return i
    return -1

def PLL2(NRZa, a, fs, baud):

    Ns = 1.0*fs/baud
    dpll = np.round(2**32 / Ns).astype(np.int32)    #increment
    ppll = -dpll    #previous state
    pll = 0 #current state

    idx = np.zeros(len(NRZa)//int(Ns)*2, dtype = np.int32)   # allocate space to save indexes    
    c = 0
    
    for n in range(1,len(NRZa)):
        if (pll < 0) and (ppll >0):
            idx[c] = n
            c = c+1
    
        if (NRZa[n] >= 0) !=  (NRZa[n-1] >=0):
            pll = np.int32(pll*a)
        
        ppll = pll
        pll = np.int32(pll+ dpll)

    return idx[:c]

def save_IQ_to_bin_file(file_name, IQ_data):
    IQ_data.tofile(file_name + ".bin")

def bits_list_to_number(bits_list):
    if bits_list == []:
        return 0

    # Convert the list of bits to a string representation
    bits_string = ''.join(str(bit) for bit in bits_list)
    
    # Convert the binary string to an integer using the int() function
    number = int(bits_string, 2)
    return number

def bits_list_to_ascii(bits_list):
    if bits_list == []:
        return 0
    # Convert the list of bits to a string representation
    bits_string = ''.join(str(bit) for bit in bits_list)

    # Convert the binary string to an integer using the int() function
    number = int(bits_string, 2)

    # Convert the integer to an ASCII character using chr()
    ascii_char = chr(number)
    
    if type(ascii_char) == str:
        return ascii_char
    else:
        return 0

def main():
    #sdr config
    Fc = 433e6
    Fs = 1024000

    info_bits = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    byte_size = 8
    decimation_factor = 3
    fft_size = 2**18
    squelch_threshold_diff = -20

    signals = []
    signals.append(get_samples_from_file('OOSK_1', bin = True, numpytype = 'complex128'))
    signals.append(get_samples_from_file('OOSK_2', bin = True, numpytype = 'complex128'))
    signals.append(get_samples_from_file('OOSK_3', bin = True, numpytype = 'complex128'))
    signals.append(get_samples_from_file('OOSK_4', bin = True, numpytype = 'complex128'))

    h1 = scipy.signal.firwin(10, Fs/decimation_factor, window = "hamming", fs = Fs, pass_zero = "lowpass")

    Fs = Fs//decimation_factor
    while True:
        for x in signals:
            simple_psd(x, 2**8, Fs, frf = Fc)
            spectogram(x, 2**8, Fs, graph = True, n_point_fft = 2**8)

            x_smoothed = scipy.signal.convolve(np.abs(x), h1)[::decimation_factor]
            x_smoothed_log = 10*np.log10(x_smoothed**2)

            if np.max(x_smoothed_log) > squelch_threshold_diff: #threshold is in db 
                
                x_demod = np.where(x_smoothed > (np.max(x_smoothed) + np.min(x_smoothed))/2, 1, -1)

                #getting NRZ signal baud rate
                x_diff = np.diff(x_demod)**2
                actual_rate = find_clock_frequency(np.abs(np.fft.fft(x_diff, fft_size)))*Fs/fft_size
                print("Baud Rate: " + str(actual_rate))

                #clock recovery
                x_normalized = x_smoothed - (np.max(x_smoothed) + np.min(x_smoothed))/2
                                                        #55331
                idx = PLL2(x_normalized, a = 0.4, fs = Fs, baud = actual_rate)   #alright this is fast!!
                bits = []
                #sample and convert to boolean
                for i in range(len(idx)):
                    if x_normalized[idx[i]] > 0:
                        bits.insert(len(bits),1)
                    if x_normalized[idx[i]] < 0:
                        bits.insert(len(bits),0)

                plt.stem(idx, x_demod[idx])
                plt.plot(x_normalized)
                plt.show()
                print(bits)

                info_index = find_bit_sequence(bits, info_bits)
                payload_byte_size = bits_list_to_number(bits[info_index-byte_size:info_index]) - 3

                if info_index != -1 and payload_byte_size > 0:
                    payload_string = ''
                    ascii_char = 0

                    for i in range(0, payload_byte_size):
                        ascii_char = bits_list_to_ascii(bits[info_index+byte_size*(2 + i):info_index+byte_size*(3 + i)])
                        if ascii_char != 0:
                            payload_string += ascii_char

                    print('Squelch Array Index in Matrix:')
                    print(info_index)

                    print('Payload Size:')
                    print(payload_byte_size)

                    print('Payload:')
                    print(payload_string)

if __name__ == '__main__':
    main()
