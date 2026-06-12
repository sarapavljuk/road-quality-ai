import os
import glob
import re
import struct
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

ACCELEROMETER_SCALE = 6.125e-5
GYROSCOPE_SCALE = 8.75e-3
MAGNETOMETER_SCALE = 1.5e-3

SCALES = {
    1: ACCELEROMETER_SCALE,
    2: GYROSCOPE_SCALE,
    3: MAGNETOMETER_SCALE
}

BYTES_PER_SAMPLE = {
    1: 6,
    2: 6,
    3: 6
}

SEGMENT_LENGTH = 3.0
MIN_SEGMENT_SAMPLES = 64


def ask_yes_no(text, default=True):
    default_hint = 'da' if default else 'ne'

    allowed_yes = ('', 'd', 'da', 'y', 'yes') if default else ('d', 'da', 'y', 'yes')
    allowed_no = ('n', 'ne', 'no') if default else ('', 'n', 'ne', 'no')

    while True:
        answer = input(f'{text} (da/ne, privzeto: {default_hint}): ').strip().lower()

        if answer in allowed_yes:
            return True

        if answer in allowed_no:
            return False

        print('Neveljaven vnos. Vpiši "da" ali "ne".')


def unstuff(data):
    result = []
    i = 0

    while i < len(data):
        if data[i] == 0xFE:
            i += 1
            if i < len(data):
                result.append(0xFE ^ data[i])
        else:
            result.append(data[i])

        i += 1

    return bytes(result)


def parse_packet(raw):
    try:
        if len(raw) < 10:
            return None

        timestamp = struct.unpack('<i', raw[3:7])[0]

        pos = 9
        chunks = []

        while pos + 3 <= len(raw):
            chunk_id = raw[pos]
            stuffed_size = struct.unpack('<H', raw[pos + 1:pos + 3])[0]

            data_start = pos + 3
            data_end = data_start + stuffed_size

            if data_end > len(raw):
                break

            chunk_data = unstuff(raw[data_start:data_end])
            chunks.append((chunk_id, chunk_data))

            pos = data_end

        if not chunks:
            return None

        return timestamp, chunks

    except Exception:
        return None


def decode_packets(raw):
    packets = []
    i = 0

    while i < len(raw) - 1:
        if raw[i] == 0xFF and raw[i + 1] == 0xFF:
            j = i + 2

            while j < len(raw) - 1:
                if raw[j] == 0xFF and raw[j + 1] == 0xFF:
                    break
                j += 1

            parsed = parse_packet(raw[i:j])

            if parsed:
                timestamp, chunks = parsed

                for chunk_id, chunk_data in chunks:
                    packets.append((timestamp, chunk_id, chunk_data))

            i = j
        else:
            i += 1

    return packets


def build_signals(packets):
    from collections import defaultdict

    grouped_by_sensor = defaultdict(list)

    for timestamp, chunk_id, chunk_data in packets:
        grouped_by_sensor[chunk_id].append((timestamp, chunk_data))

    results = {}

    for chunk_id, entries in grouped_by_sensor.items():
        if len(entries) < 2:
            continue

        bytes_per_sample = BYTES_PER_SAMPLE.get(chunk_id, 6)

        samples = []
        timestamps = [timestamp for timestamp, _ in entries]

        for timestamp, chunk_data in entries:
            n = len(chunk_data) // bytes_per_sample

            for k in range(n):
                chunk = chunk_data[k * bytes_per_sample:(k + 1) * bytes_per_sample]

                if len(chunk) == 6:
                    samples.append(struct.unpack('<3h', chunk))

        if not samples:
            continue

        differences = [
            timestamps[i + 1] - timestamps[i]
            for i in range(len(timestamps) - 1)
            if timestamps[i + 1] - timestamps[i] > 0
        ]

        if not differences:
            continue

        average_period_ms = sum(differences) / len(differences)
        samples_per_packet = len(samples) / len(entries)
        fs = samples_per_packet / (average_period_ms / 1000.0)

        matrix = np.array(samples, dtype=float)

        if chunk_id in SCALES:
            matrix *= SCALES[chunk_id]

        results[chunk_id] = (fs, matrix)

    return results


def remove_dc(signal):
    return signal - np.mean(signal, axis=0)


def filter_noise(signal, fs, low_cutoff=0.5, order=4):
    min_samples = 3 * order * 2

    if fs <= 2.0 or signal.shape[0] < min_samples:
        return signal

    nyquist = fs / 2.0
    high_cutoff = nyquist * 0.8

    if low_cutoff >= high_cutoff:
        return signal

    try:
        sos = scipy_signal.butter(
            order,
            [low_cutoff / nyquist, high_cutoff / nyquist],
            btype='bandpass',
            output='sos'
        )

        output = np.zeros_like(signal)

        for axis_index in range(signal.shape[1]):
            output[:, axis_index] = scipy_signal.sosfiltfilt(
                sos,
                signal[:, axis_index]
            )

        return output

    except Exception:
        return signal


def split_into_segments(signal, fs):
    segment_size = int(round(SEGMENT_LENGTH * fs))

    if signal.shape[0] <= segment_size:
        return [signal]

    segments = []
    start = 0

    while start + segment_size <= signal.shape[0]:
        end = start + segment_size
        segments.append(signal[start:end])
        start += segment_size

    return segments


def label_from_filename(filename):
    return re.sub(r'\d+$', '', filename)


def main():
    use_segmentation = ask_yes_no(
        'Želiš vključiti segmentiranje podatkov?',
        default=False
    )

    folder = os.path.dirname(os.path.abspath(__file__))

    bin_files = sorted(
        glob.glob(os.path.join(folder, '*.bin')) +
        glob.glob(os.path.join(folder, '*.BIN'))
    )

    if not bin_files:
        print('Ni najdenih .bin/.BIN datotek v mapi:', folder)
        return

    output_folder = os.path.join(folder, 'out_acc_id1')
    os.makedirs(output_folder, exist_ok=True)

    print(f'Najdenih {len(bin_files)} .bin/.BIN datotek.')
    print(f'Izhod: {output_folder}')
    print('Uporabljamo samo senzor ID=1 kot accelerometer / pospeškometer')
    print('Brez glajenja in brez amplitudne normalizacije')
    print(f'Segmentiranje: {"vključeno" if use_segmentation else "izključeno"}')
    print(f'Minimalna dolžina segmenta: {MIN_SEGMENT_SAMPLES} vzorcev\n')

    for path in bin_files:
        name = os.path.splitext(os.path.basename(path))[0]
        label = label_from_filename(name)

        print(f'Obdelujem: {os.path.basename(path)} | label={label}')

        with open(path, 'rb') as f:
            raw = f.read()

        packets = decode_packets(raw)
        results = build_signals(packets)

        if not results:
            print(' ✗ Ni bilo mogoče dekodirati signalov.\n')
            continue

        print(f' Najdeni senzorji: {list(results.keys())}')

        if 1 not in results:
            print(' ✗ Senzor ID=1 ni najden.\n')
            continue

        fs, matrix = results[1]

        if matrix.shape[0] < MIN_SEGMENT_SAMPLES or fs < 2.0:
            print(
                f' ✗ Premalo podatkov: {matrix.shape[0]} vzorcev, fs={fs:.2f} Hz\n'
            )
            continue

        duration = matrix.shape[0] / fs

        print(
            f' Senzor ID=1 | fs={fs:.1f} Hz | '
            f'vzorcev={matrix.shape[0]} | trajanje={duration:.2f}s'
        )

        signal = remove_dc(matrix)
        signal = filter_noise(signal, fs)

        if use_segmentation:
            segments = split_into_segments(signal, fs)
        else:
            segments = [signal]

        saved_count = 0

        for index, segment in enumerate(segments, start=1):
            if segment.shape[0] < MIN_SEGMENT_SAMPLES:
                print(
                    f' Preskočeno: segment {index} je prekratek '
                    f'({segment.shape[0]} vzorcev)'
                )
                continue

            df = pd.DataFrame(segment, columns=['X', 'Y', 'Z'])
            df['label'] = label

            if use_segmentation and len(segments) > 1:
                output_name = f'{name}_id1_seg{index}.csv'
            else:
                output_name = f'{name}_id1.csv'

            output_path = os.path.join(output_folder, output_name)

            df.to_csv(output_path, index=False)

            saved_count += 1

            print(
                f' → {output_name} '
                f'({segment.shape[0]} vzorcev, label={label})'
            )

        if saved_count == 0:
            print(' ✗ Noben segment ni bil shranjen.')

        print()


if __name__ == '__main__':
    main()