import struct
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from STM32DataLoggerDecoder import STM32DataLoggerDecoder

# Rezolucije senzorjev glede na ID
SENSOR_RESOLUTION = {
    1: 8.75e-3,    # Žiroskop [°/s]
    2: 6.125e-5,   # Pospeškomter [g]
    3: 1.5e-3,     # Magnetometer [Gauss]
}
SENSOR_BYTES_PER_SAMPLE = {1: 2, 2: 2, 3: 2} 

SENSOR_LABELS = {
    1: ("Gyroscope", "rotational speed (degrees per second)", f"100 Hz, res=8.75e-3 D/s"),
    2: ("Accelerometer", "acceleration (G-force)", f"25 Hz, res=6.125e-5 g"),
    3: ("Magnetometer", "magnetic field (Gauss)", f"10 Hz, res=1.5e-3 Gauss"),
}

@dataclass
class Paket:
    id: int
    ts: float
    data: np.ndarray


def dekodiraj_datoteko(filename) -> list:
    """Razširi STM32DataLoggerDecoder, da vrne seznam Paket objektov."""
    dec = STM32DataLoggerDecoder(filename)

    with open(filename, 'rb') as f:
        content = f.read()

    raw_chunks = content.split(dec.sync_marker)
    paketi = []

    for raw_data in raw_chunks:
        if len(raw_data) < 5:
            continue

        payload = dec.unstuff_payload(raw_data[1:])
        if len(payload) < 6:
            continue


        ts_ms = struct.unpack('<I', payload[0:4])[0]
        ts = ts_ms / 1000.0

        chunks_area = payload[6:-2] if len(payload) > 8 else payload[6:]
        pos = 0

        while pos < len(chunks_area):
            if pos + 4 > len(chunks_area):
                break
            try:
                sensor_id = chunks_area[pos]
                chunk_size = struct.unpack('<H', chunks_area[pos+1:pos+3])[0] + 1
                raw = bytes(chunks_area[pos+4:pos+4+chunk_size])
                data = np.frombuffer(raw, dtype=np.int16)
                paketi.append(Paket(id=sensor_id, ts=ts, data=data))
                pos += 4 + chunk_size
            except Exception:
                break

    return paketi


def sestavi_podatke(seznam_paketov: list):
    """Sestavi seznam paketov v Numpy matriko N x 3 in izračuna Fvz."""
    if not seznam_paketov:
        return 0.0, np.array([])


    timestamps = [p.ts for p in seznam_paketov]
    T_paketi = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    T_avg = np.mean(T_paketi) if T_paketi else 1.0

    sensor_id = seznam_paketov[0].id
    bytes_per_val = SENSOR_BYTES_PER_SAMPLE.get(sensor_id, 2)
    resolution = SENSOR_RESOLUTION.get(sensor_id, 1.0)
    axes = 3

    Nvz_list = [len(p.data) // axes for p in seznam_paketov]
    Nvz_avg = np.mean(Nvz_list)
    Fvz = Nvz_avg / T_avg if T_avg > 0 else 0.0


    vsi = []
    for p in seznam_paketov:
        vzorci = len(p.data) // axes
        mat = p.data[:vzorci * axes].reshape(vzorci, axes).astype(float) * resolution
        vsi.append(mat)

    signal = np.vstack(vsi) if vsi else np.array([])
    return Fvz, signal


def prikazi_signal(signal: np.ndarray, naslov: str = "", startInd: int = None, endInd: int = None,
                   ylabel: str = "vrednost", Fvz: float = None):
    """Prikaže signal (N x 3) kot graf z osmi X, Y, Z."""
    if startInd is not None or endInd is not None:
        s = startInd if startInd is not None else 0
        e = endInd if endInd is not None else len(signal)
        signal = signal[s:e]

    N = len(signal)
    if Fvz and Fvz > 0:
        t = np.arange(N) / Fvz
        xlabel = "čas (s)"
    else:
        t = np.arange(N)
        xlabel = "vzorec"

    fig, ax = plt.subplots(figsize=(12, 4))
    koordinate = ['x', 'y', 'z']
    for i, ime in enumerate(koordinate):
        if signal.ndim == 2 and signal.shape[1] > i:
            ax.plot(t, signal[:, i], label=ime)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if naslov:
        ax.set_title(naslov)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    DATOTEKA = "lezeci4.BIN"


    vsi_paketi = dekodiraj_datoteko(DATOTEKA)
    print(f"Skupaj prebranih paketov: {len(vsi_paketi)}")


    for sensor_id, (ime, enota, info) in SENSOR_LABELS.items():
        paketi = [p for p in vsi_paketi if p.id == sensor_id]
        if not paketi:
            print(f"Ni podatkov za {ime} (id={sensor_id})")
            continue


        Fvz, signal = sestavi_podatke(paketi)
        print(f"{ime}: {len(signal)} vzorcev, Fvz = {Fvz:.1f} Hz")

        naslov_cel = f"{ime}\n(Fvz = {Fvz:.1f} Hz, {info})"


        prikazi_signal(signal, naslov=naslov_cel, ylabel=enota, Fvz=Fvz)


    plt.show()
