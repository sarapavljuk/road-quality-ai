import os
import glob
import re
import struct
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

GYROSCOPE_SCALE = 8.75e-3
ACCELEROMETER_SCALE = 6.125e-5
MAGNETOMETER_SCALE = 1.5e-3

SCALES = {1: GYROSCOPE_SCALE, 2: ACCELEROMETER_SCALE, 3: MAGNETOMETER_SCALE}
BYTES_PER_SAMPLE = {1: 6, 2: 6, 3: 6}

SEGMENT_LENGTH = 5.0
DOUBLE_THRESHOLD = 10.0
SMOOTHING_WINDOW = 5


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
    """
    Sprejme surove bajte enega paketa.
    Vrne (timestamp_ms, [(chunk_id, chunk_data_bytes), ...]) ali None.
    """
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
    """
    Vrne list (timestamp_ms, chunk_id, chunk_data) za vse veljavne pakete.
    """
    packets = []
    i = 0
    while i < len(raw) - 1:
        if raw[i] == 0xFF and raw[i + 1] == 0xFF:
            j = i + 2
            while j < len(raw) - 1:
                if raw[j] == 0xFF and raw[j + 1] == 0xFF:
                    break
                j += 1
            paket_raw = raw[i:j]
            parsed = parse_packet(paket_raw)
            if parsed:
                ts, chunks = parsed
                for cid, cdata in chunks:
                    packets.append((ts, cid, cdata))
            i = j
        else:
            i += 1
    return packets



def assemble_signals(packets):
    """
    Returns dict: {sensor_id: (fs_Hz, matrix_Nx3)}
    """
    from collections import defaultdict

    by_sensor = defaultdict(list)
    for ts, cid, cdata in packets:
        by_sensor[cid].append((ts, cdata))

    results = {}
    for cid, sensor_data in by_sensor.items():
        if len(sensor_data) < 2:
            continue

        bytes_per_sample = BYTES_PER_SAMPLE.get(cid, 6)
        samples = []
        timestamps = [ts for ts, _ in sensor_data]

        for ts, cdata in sensor_data:
            n = len(cdata) // bytes_per_sample
            for k in range(n):
                chunk = cdata[k * bytes_per_sample:(k + 1) * bytes_per_sample]
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
        samples_per_packet = len(samples) / len(sensor_data)
        fs = samples_per_packet / (average_period_ms / 1000.0)

        matrix = np.array(samples, dtype=float)
        if cid in SCALES:
            matrix *= SCALES[cid]

        results[cid] = (fs, matrix)

    return results



def remove_dc(sig):
    return sig - np.mean(sig, axis=0)



def center_signal(sig):
    center = (np.max(sig, axis=0) + np.min(sig, axis=0)) / 2.0
    return sig - center



def filter_noise(sig, fs, low=0.5, order=4):
    min_samples = 3 * order * 2
    if fs <= 2.0 or sig.shape[0] < min_samples:
        return sig

    nyq = fs / 2.0
    high = nyq * 0.8
    low = max(low, 0.01)

    if low >= high or high >= nyq:
        high = nyq * 0.99
    if low >= high:
        return sig

    try:
        sos = scipy_signal.butter(order, [low / nyq, high / nyq], btype='bandpass', output='sos')
        out = np.zeros_like(sig)
        for axis_idx in range(sig.shape[1]):
            out[:, axis_idx] = scipy_signal.sosfiltfilt(sos, sig[:, axis_idx])
        return out
    except Exception:
        return sig



def smooth_signal(sig, window=SMOOTHING_WINDOW):
    if window < 2 or sig.shape[0] < window:
        return sig

    kernel = np.ones(window, dtype=float) / window
    out = np.zeros_like(sig)

    for axis_idx in range(sig.shape[1]):
        out[:, axis_idx] = np.convolve(sig[:, axis_idx], kernel, mode='same')

    return out



def ask_yes_no(text, default=True):
    default_hint = 'da' if default else 'ne'
    allowed_yes = ('', 'd', 'da', 'y', 'yes') if default else ('d', 'da', 'y', 'yes')
    allowed_no = ('n', 'ne', 'no') if default else ('', 'n', 'ne', 'no')

    while True:
        answer = input(f'{text} (da/ne, default: {default_hint}): ').strip().lower()
        if answer in allowed_yes:
            return True
        if answer in allowed_no:
            return False
        print('Neveljaven vnos. Vpiši "da" ali "ne".')



def ask_for_smoothing():
    return ask_yes_no(
        f'Želiš vključiti glajenje podatkov? Okno glajenja = {SMOOTHING_WINDOW}',
        default=True,
    )



def ask_for_segmentation():
    return ask_yes_no(
        f'Želiš vključiti segmentiranje podatkov? Segment = {SEGMENT_LENGTH}s, dvojni prag = {DOUBLE_THRESHOLD}s',
        default=True,
    )



def ask_for_labeling():
    return ask_yes_no(
        'Želiš vključiti labeling iz imena .bin datoteke?',
        default=True,
    )



def normalize_amplitude(sig):
    max_val = np.max(np.abs(sig))
    return sig / max_val if max_val != 0 else sig



def split_into_segments(sig, fs):
    n = sig.shape[0]
    t = n / fs
    n_seg = int(round(SEGMENT_LENGTH * fs))

    if t < SEGMENT_LENGTH:
        return [sig]
    elif t < DOUBLE_THRESHOLD:
        return [sig[:n_seg]]
    else:
        mid = n // 2
        s2 = max(0, mid - n_seg // 2)
        e2 = min(n, s2 + n_seg)
        if e2 - s2 < n_seg:
            s2 = max(0, e2 - n_seg)
        return [sig[:n_seg], sig[s2:e2]]



def label_from_filename(filename):
    """
    - slaba_cesta  -> slaba_cesta
    - slaba_cesta2 -> slaba_cesta
    - slaba_cesta15 -> slaba_cesta
    """
    return re.sub(r'\d+$', '', filename)



def main():
    use_smoothing = ask_for_smoothing()
    use_segmentation = ask_for_segmentation()
    use_labeling = ask_for_labeling()

    folder = os.path.dirname(os.path.abspath(__file__))
    bin_files = sorted(
        glob.glob(os.path.join(folder, '*.bin')) +
        glob.glob(os.path.join(folder, '*.BIN'))
    )

    if not bin_files:
        print('Ni najdenih .bin/.BIN datotek v mapi:', folder)
        return

    output_folder = os.path.join(folder, 'out')
    os.makedirs(output_folder, exist_ok=True)

    print(f'Najdenih {len(bin_files)} .bin/.BIN datotek.')
    print(f'Izhod: {output_folder}')
    print(f'Glajenje podatkov: {"vključeno" if use_smoothing else "izključeno"}')
    print(f'Segmentiranje podatkov: {"vključeno" if use_segmentation else "izključeno"}')
    print(f'Labeling podatkov: {"vključen" if use_labeling else "izključen"}\n')

    for path in bin_files:
        name = os.path.splitext(os.path.basename(path))[0]
        label = label_from_filename(name) if use_labeling else None

        if use_labeling:
            print(f'Obdelujem: {os.path.basename(path)} | label={label}')
        else:
            print(f'Obdelujem: {os.path.basename(path)}')

        with open(path, 'rb') as f:
            raw_data = f.read()

        packets = decode_packets(raw_data)
        results = assemble_signals(packets)

        if not results:
            print(' ✗ Ni bilo mogoče dekodirati signalov.\n')
            continue

        for cid, (fs, matrix) in results.items():
            if cid == 3:
                print(f' Preskakujem senzor ID={cid} (magnetometer)')
                continue
            if cid not in (1, 2):
                print(f' Preskakujem senzor ID={cid} (neznan tip)')
                continue
            if matrix.shape[0] < 10 or fs < 2.0:
                print(f' Preskakujem senzor ID={cid} (premalo podatkov: {matrix.shape[0]} vzorcev, fs={fs:.2f} Hz)')
                continue

            t = matrix.shape[0] / fs
            print(f' Senzor ID={cid} | fs={fs:.1f} Hz | vzorcev={matrix.shape[0]} | trajanje={t:.2f}s')

            sig = remove_dc(matrix)
            sig = center_signal(sig)
            sig = filter_noise(sig, fs)
            if use_smoothing:
                sig = smooth_signal(sig)
            sig = normalize_amplitude(sig)

            segments = split_into_segments(sig, fs) if use_segmentation else [sig]

            for idx, segment in enumerate(segments, start=1):
                df = pd.DataFrame(segment, columns=['X', 'Y', 'Z'])
                if use_labeling:
                    df['label'] = label

                if use_segmentation and len(segments) > 1:
                    output_name = f'{name}_id{cid}_seg{idx}.csv'
                else:
                    output_name = f'{name}_id{cid}.csv'

                output_path = os.path.join(output_folder, output_name)
                df.to_csv(output_path, index=False)

                if use_labeling:
                    print(f' → {output_name} ({segment.shape[0]} vzorcev, label={label})')
                else:
                    print(f' → {output_name} ({segment.shape[0]} vzorcev)')

        print()


if __name__ == '__main__':
    main()
