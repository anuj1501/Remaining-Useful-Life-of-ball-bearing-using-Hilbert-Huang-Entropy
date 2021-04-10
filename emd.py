from scipy.interpolate import interp1d
from numpy import array, sign, zeros
from scipy import integrate
from sklearn.metrics import auc
from scipy.signal import hilbert
from hht.emd import EMD
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from scipy import stats
import os


def preprocessor(folder, bearing_num):

    path = os.path.join(str(folder),
                        str("Bearing" + str(bearing_num))
                        )

    files = os.listdir(path)

    bearing = []

    for i, file_name in enumerate(files):
        # if i % 200==0:
        #print("executing file {}/{}".format(i+1, len(files)))

        data = pd.read_csv('{}/{}'.format(path, file_name), names=[
            'hour', 'minute', 'seconds', 'milliseconds', 'horizontal acceleration', 'vertical acceleration'])

        # print(data.tail())

        x = data["horizontal acceleration"]

        hori_acc = x[x.between(x.quantile(.20), x.quantile(.80))]

        time = np.array(data["milliseconds"])

        time = time - time[0]

        decomposer = EMD(hori_acc)

        IMFs = decomposer.decompose()

        # t = time/1000
        if i < 0:
            nimfs = IMFs.shape[0]
            #print(nimfs, IMFs.shape[1])
            plt.figure(figsize=(12, 9))
            plt.subplot(nimfs+1, 1, 1)
            plt.plot(hori_acc, 'r')
            for n in range(nimfs):
                plt.subplot(nimfs+1, 1, n+2)
                plt.plot(IMFs[n], 'g')
                #plt.ylabel("eIMF %i" %(n+1))
                plt.locator_params(axis='y', nbins=5)
            plt.show()

        integrals = []

        for i in range(len(IMFs)):

            # print(np.mean(IMFs[i]))

            analytical_signal = hilbert(IMFs[i])

            real_part = np.abs(np.real(analytical_signal))
            # real_part = np.imag(analytical_signal)

            # integral = auc(np.arange(1, 2561), real_part)

            integral = (integrate.trapz(real_part, dx=1))

            # integral = np.trapz(real_part)

            integrals.append(abs(integral))

        HHEs = []

        for i in range(len(integrals)):

            pi = (integrals[i] / sum(integrals))

            HHE = pi * (np.log(pi) / np.log(1.47))

            HHEs.append(HHE)

        Final_output = sum(HHEs)

        Final_output = (Final_output * -1)

        # print(Final_output)

        bearing.append(Final_output)

    # print(bearing)

    m = len(bearing)
    # print(m)
    b1 = np.array(bearing[:int(0.90*m)])
    b2 = np.array(bearing[int(0.90*m):])

    # t1 = np.linspace(0, 0.5, len(b2))
    # t1 = t1[::-1]
    t1 = np.random.uniform(low=0, high=0.5, size=(len(b2),))
    t1.sort()
    t1 = t1[::-1]

    t2 = np.multiply(t1, b2)

    final_bearing = np.concatenate((b1, t2))

    plt.plot(final_bearing)
    plt.show()

    x = max(final_bearing)

    reversal = final_bearing.copy()

    reversal = [((i * -1) + max(reversal)) for i in reversal]

    plt.plot(reversal)
    plt.show()

    bearing_reversal = np.array(reversal)

    q_u = zeros(bearing_reversal.shape)

    # Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.

    u_x = [0, ]
    u_y = [bearing_reversal[0], ]

    # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1, len(bearing_reversal)-1, 4):
        if (sign(bearing_reversal[k]-bearing_reversal[k-1]) == 1) and (sign(bearing_reversal[k]-bearing_reversal[k+1]) == 1):
            u_x.append(k)
            u_y.append(bearing_reversal[k])

    # Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.

    u_x.append(len(bearing_reversal)-1)
    u_y.append(bearing_reversal[-1])

    u_p = interp1d(u_x, u_y, kind='cubic',
                   bounds_error=False, fill_value=0.0)

    # Evaluate each model over the domain of (s)
    for k in range(0, len(bearing_reversal)):
        q_u[k] = u_p(k)

    # Plot everything

    maxy = max(q_u)
    maxx = np.argmax(q_u)
    #print("bearing rev",bearing_reversal)
    #print("q", maxx,maxy)
    plt.plot(np.arange(len(bearing_reversal)), bearing_reversal, 'b')
    plt.plot(np.arange(len(q_u)), q_u, 'r')
    plt.plot(maxx, maxy, marker='s', markersize=15, color='g', label="FT")
    plt.grid(True)
    plt.ylim((0, 6))
    plt.legend()
    plt.show()

    return bearing_reversal


if __name__ == "__main__":
    training_set = ["1_1", "1_2", "2_1", "2_2", "3_1", "3_2"]
    for i, bearing_set in enumerate(training_set):
        answer = preprocessor(
            "ieee-phm-2012-data-challenge-dataset/Learning_set", bearing_set)
        pd.DataFrame(answer).to_csv(
            "foo{}.csv".format(i), header=None, index=None)
