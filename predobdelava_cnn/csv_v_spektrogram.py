import os
import glob
import numpy as np
from scipy.signal import stft
from PIL import Image

FS = 100

NPERSEG = 64
NOVERLAP = 48
NFFT = 128

INPUT_FOLDER = "dataset_csv"
OUTPUT_FOLDER = "dataset_spektrogrami_v7_magnitude"

CLASSES = [
    "dobra_cesta",
    "slaba_cesta",
    "luknja",
    "lezeci_policaj"
]


def spectrogram_to_image(zxx):
    zxx = np.abs(zxx)
    zxx = np.log1p(zxx)

    zxx = zxx - np.min(zxx)

    max_val = np.max(zxx)

    if max_val > 0:
        zxx = zxx / max_val

    zxx = zxx * 255

    return zxx.astype(np.uint8)


def create_stft(signal):
    if len(signal) < 64:
        return None

    nperseg = min(NPERSEG, len(signal))
    noverlap = min(NOVERLAP, nperseg - 1)
    nfft = max(NFFT, nperseg)

    f, t, zxx = stft(
        signal,
        fs=FS,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft
    )

    return spectrogram_to_image(zxx)


def csv_to_rgb_spectrogram(csv_path, output_path):
    data = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1, 2)
    )

    if data.ndim != 2 or data.shape[1] < 3:
        print(f"Preskočeno: {csv_path}")
        return

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    sx = create_stft(x)
    sy = create_stft(y)
    sm = create_stft(magnitude)

    if sx is None or sy is None or sm is None:
        print(f"Prekratek signal: {csv_path}")
        return

    min_h = min(sx.shape[0], sy.shape[0], sm.shape[0])
    min_w = min(sx.shape[1], sy.shape[1], sm.shape[1])

    sx = sx[:min_h, :min_w]
    sy = sy[:min_h, :min_w]
    sm = sm[:min_h, :min_w]

    rgb = np.stack([sx, sy, sm], axis=2)

    image = Image.fromarray(rgb)

    image = image.resize((224, 224), Image.NEAREST)

    image.save(output_path)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    total = 0

    for class_name in CLASSES:
        input_class_folder = os.path.join(INPUT_FOLDER, class_name)
        output_class_folder = os.path.join(OUTPUT_FOLDER, class_name)

        os.makedirs(output_class_folder, exist_ok=True)

        csv_files = glob.glob(os.path.join(input_class_folder, "*id1*.csv"))

        print(f"{class_name}: {len(csv_files)} CSV datotek")

        for csv_path in csv_files:
            name = os.path.splitext(os.path.basename(csv_path))[0]

            output_path = os.path.join(
                output_class_folder,
                name + "_spektrogram.png"
            )

            csv_to_rgb_spectrogram(csv_path, output_path)

            total += 1

    print("\nPretvorba končana.")
    print(f"Obdelanih datotek: {total}")
    print(f"Izhodna mapa: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()