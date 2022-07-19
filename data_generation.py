import os
import uuid
import numpy as np
import argparse
import json
from pathlib import Path

# Training settings
parser = argparse.ArgumentParser(description="First binarized DoA estimation example")
parser.add_argument(
    "--data-size",
    type=int,
    default=1000000,
    metavar="N",
    help="data size in samples (default: 1e6)",
)
parser.add_argument(
    "--sources",
    type=int,
    default=1,
    metavar="N",
    help="Number of sources (default: 1)",
)
parser.add_argument(
    "--realizations",
    type=int,
    default=1,
    metavar="N",
    help="Number of realizations (default: 1)",
)
parser.add_argument(
    "--array-elements",
    type=int,
    default=1024,
    metavar="N",
    help="Array elements (default: 1024)",
)
parser.add_argument(
    "--snr",
    type=int,
    default=1000,
    metavar="N",
    help="Signal to noise ratio (default: 1000)",
)
parser.add_argument(
    "--angular-bins",
    type=int,
    default=90,
    metavar="N",
    help="Angular bins (default: 90)",
)


def array_response_vector(array, theta):
    N = array.shape
    v = np.exp(1j * 2 * np.pi * array * np.sin(theta))
    return v / np.sqrt(N)


def generate_single_data(L, N, snr, numrealization):
    array = np.linspace(0, (N - 1) / 2, N)
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


def generate_all_data(L, N, snr, numrealization, angular_bins, datapoints, path):
    bins = angular_bins
    Hout = np.zeros((datapoints, 2 * N * numrealization), dtype=np.float16)
    Thetaout = np.zeros((datapoints), dtype=np.int16)

    for i in range(datapoints):
        Thetas, H = generate_single_data(L, N, snr, numrealization)

        # dThetas = np.zeros((bins), dtype=np.int8)
        idx = np.floor(((Thetas + np.pi / 2) / np.pi) * (bins)).astype(np.int16)
        # dThetas[idx] = 1

        flatH = H.flatten().T

        binH = np.column_stack((flatH.real, flatH.imag)).flatten().astype(np.float16)

        Hout[i] = binH
        Thetaout[i] = idx  # dThetas

    with open(f"{path}/signal.npy", "wb") as signal, open(
        f"{path}/label.npy", "wb"
    ) as label:
        np.save(signal, Hout)
        np.save(label, Thetaout)


def main():
    args = parser.parse_args()

    name = uuid.uuid4().hex

    Path("data").mkdir(parents=True, exist_ok=True)

    path = f"data/{name}"
    os.mkdir(path)

    with open(f"{path}/arguments.json", "w") as args_json:
        json.dump(args.__dict__, args_json, indent=2)

    generate_all_data(
        args.sources,
        args.array_elements,
        args.snr,
        args.realizations,
        args.angular_bins,
        args.data_size,
        path,
    )


if __name__ == "__main__":
    main()
