import numpy as np


def array_response_vector(array, theta):
    N = array.shape
    v = np.exp(1j * 2 * np.pi * array * np.sin(theta))
    return v / np.sqrt(N)


L = 1  # number of sources
N = 64  # number of ULA elements
snr = 10000  # signal to noise ratio
numrealization = 64  # number of realizations

array = np.linspace(0, (N - 1) / 2, N)


def generate_single_data(L, N, snr, numrealization):
    Thetas = np.pi * (np.random.rand(L) - 1 / 2)  # random source directions
    Alphas = np.random.randn(L) + np.random.randn(L) * 1j  # random source powers
    Alphas = np.sqrt(1 / 2) * Alphas

    H = np.zeros((N, numrealization)) + 1j * np.zeros((N, numrealization))

    for iter in range(numrealization):
        htmp = np.zeros(N)
        for i in range(L):
            pha = np.exp(1j * 2 * np.pi * np.random.rand(1))
            htmp = htmp + pha * Alphas[i] * array_response_vector(array, Thetas[i])
        wgn = np.sqrt(0.5 / snr) * (np.random.randn(N) + np.random.randn(N) * 1j)
        H[:, iter] = htmp + wgn
    return Thetas, H


def generate_all_data(L, N, snr, numrealization, datapoints):
    bins = 90
    Hout = np.zeros((datapoints, 2 * N * numrealization), dtype=np.int8)
    Thetaout = np.zeros((datapoints, bins), dtype=np.int8)

    for i in range(datapoints):
        Thetas, H = generate_single_data(L, N, snr, numrealization)

        dThetas = np.zeros((bins), dtype=np.int8)
        idx = np.floor(((Thetas + np.pi / 2) / np.pi) * (bins)).astype(np.uint32)
        dThetas[idx] = 1

        flatH = H.flatten().T

        binH = np.sign(np.column_stack((flatH.real, flatH.imag)).flatten()).astype(
            np.int8
        )

        Hout[i] = binH
        Thetaout[i] = dThetas

    with open("signal.npy", "wb") as signal, open("label.npy", "wb") as label:
        np.save(signal, Hout)
        np.save(label, Thetaout)


# generate_all_data(L, N, snr, numrealization, 10000)


with open("signal.npy", "rb") as f:
    a = np.load(f)
    print(a.shape)


with open("label.npy", "rb") as f:
    a = np.load(f)
    print(a.shape)
