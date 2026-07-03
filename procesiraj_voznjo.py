import os
import struct
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import kurtosis, skew


SENSOR_RESOLUTION = {1: 8.75e-3, 2: 6.125e-5, 3: 1.5e-3}
SENSOR_BYTES_PER_SAMPLE = {1: 2, 2: 2, 3: 2}

@dataclass
class Paket:
    id: int
    ts: float
    data: np.ndarray

class STM32DataLoggerDecoder:
    def __init__(self):
        self.sync_marker = b'\xFF\xFF'

    def unstuff_payload(self, data):
        unstuffed = bytearray()
        i = 0
        while i < len(data):
            if data[i] == 0xFE:
                if i + 1 < len(data):
                    unstuffed.append(data[i+1] ^ 0xFE)
                    i += 2
                else: break
            else:
                unstuffed.append(data[i])
                i += 1
        return bytes(unstuffed)

def preberi_bin_v_pakete(filename) -> list:
    dec = STM32DataLoggerDecoder()
    if not os.path.exists(filename):
        print(f"Datoteka {filename} ne obstaja!")
        return []

    with open(filename, 'rb') as f:
        content = f.read()

    raw_chunks = content.split(dec.sync_marker)
    paketi = []

    for raw_data in raw_chunks:
        if len(raw_data) < 5: continue
        payload = dec.unstuff_payload(raw_data[1:])
        if len(payload) < 6: continue

        ts_ms = struct.unpack('<I', payload[0:4])[0]
        ts = ts_ms / 1000.0  
        chunks_area = payload[6:-2] if len(payload) > 8 else payload[6:]
        pos = 0

        while pos < len(chunks_area):
            if pos + 4 > len(chunks_area): break
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


def ustvari_zvezni_signal(paketi, id_senzorja):
    senzor_paketi = [p for p in paketi if p.id == id_senzorja]
    if not senzor_paketi: return np.array([])

    resolution = SENSOR_RESOLUTION.get(id_senzorja, 1.0)
    all_rows = []

    for i, p in enumerate(senzor_paketi):
        vzorci = len(p.data) // 3
        mat = p.data[:vzorci * 3].reshape(vzorci, 3).astype(float) * resolution
        
        if i < len(senzor_paketi) - 1:
            next_ts = senzor_paketi[i+1].ts
        else:
            next_ts = p.ts + (p.ts - senzor_paketi[i-1].ts) if i > 0 else p.ts + 0.1
            
        t_arr = np.linspace(p.ts, next_ts, vzorci, endpoint=False)
        for t_val, osi in zip(t_arr, mat):
            all_rows.append([t_val, osi[0], osi[1], osi[2]])
            
    return np.array(all_rows)

def izracunaj_znacilke_policaji(data, prefiks):
    z = {}
    if data is None or len(data) == 0:
        osi = ['x', 'y', 'z']
        if "acc" in prefiks: osi.append('mag')
        for os in osi:
            z[f'{prefiks}_{os}_mean'] = 0.0
            z[f'{prefiks}_{os}_std'] = 0.0
            z[f'{prefiks}_{os}_max'] = 0.0
            z[f'{prefiks}_{os}_ptp'] = 0.0  
        return z

    osi_imena = ['x', 'y', 'z']
    if "acc" in prefiks:
        mag = np.sqrt(data[:, 1]**2 + data[:, 2]**2 + data[:, 3]**2)
        data = np.column_stack((data, mag))
        osi_imena.append('mag')

    for i, os in enumerate(osi_imena):
        stolpec = data[:, i + 1] 
        z[f'{prefiks}_{os}_mean'] = np.mean(stolpec)
        z[f'{prefiks}_{os}_std'] = np.std(stolpec)
        z[f'{prefiks}_{os}_max'] = np.max(stolpec)
        z[f'{prefiks}_{os}_ptp'] = np.ptp(stolpec) 
    return z

def generiraj_csv_policaji(paketi, izhodni_csv, okno_dolzina=5.0, korak=1.0):
    acc_data = ustvari_zvezni_signal(paketi, 2)
    gyro_data = ustvari_zvezni_signal(paketi, 1)

    if len(acc_data) == 0:
        print("Ni Accel podatkov za policaje!")
        return

    start_time = acc_data[0, 0]
    end_time = acc_data[-1, 0]
    okna_features = []
    trenutni_cas = start_time

    while trenutni_cas <= end_time - (okno_dolzina * 0.5): 
        konec_okna = trenutni_cas + okno_dolzina
        acc_okno = acc_data[(acc_data[:, 0] >= trenutni_cas) & (acc_data[:, 0] < konec_okna)]
        gyro_okno = gyro_data[(gyro_data[:, 0] >= trenutni_cas) & (gyro_data[:, 0] < konec_okna)] if len(gyro_data) > 0 else []

        zn_acc = izracunaj_znacilke_policaji(acc_okno, "acc")
        zn_gyro = izracunaj_znacilke_policaji(gyro_okno, "gyro")

        vrstica = {**zn_acc, **zn_gyro}
        vrstica['cas_okna_start'] = trenutni_cas - start_time 
        okna_features.append(vrstica)
        trenutni_cas += korak

    df = pd.DataFrame(okna_features)
    df.to_csv(izhodni_csv, index=False)
    print(f"Dataset za policaje: {izhodni_csv}")


def sestavi_podatke_cesta(seznam_paketov):
    if not seznam_paketov: return 0.0, np.array([])

    timestamps = [p.ts for p in seznam_paketov]
    T_paketi = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    T_avg = np.mean(T_paketi) if T_paketi else 1.0

    sensor_id = seznam_paketov[0].id
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

def izracunaj_znacilke_cesta(okno_vzorcev, predpona=""):
    znacilke = {}
    osi = ['X', 'Y', 'Z']
    if len(okno_vzorcev) == 0: return {}

    for i_os, oznaka_osi in enumerate(osi):
        podatki_osi = okno_vzorcev[:, i_os]
        polno_ime = f"{predpona}_{oznaka_osi}"
        
        znacilke[f"{polno_ime}_mean"] = np.mean(podatki_osi)
        znacilke[f"{polno_ime}_std"] = np.std(podatki_osi)
        znacilke[f"{polno_ime}_min"] = np.min(podatki_osi)
        znacilke[f"{polno_ime}_max"] = np.max(podatki_osi)
        znacilke[f"{polno_ime}_rms"] = np.sqrt(np.mean(podatki_osi**2))
        znacilke[f"{polno_ime}_ptp"] = np.ptp(podatki_osi)
        znacilke[f"{polno_ime}_kurtosis"] = kurtosis(podatki_osi, fisher=True)
        znacilke[f"{polno_ime}_skew"] = skew(podatki_osi)
        
    return znacilke

def generiraj_csv_cesta(paketi, izhodni_csv, okno_sekunde=5.0):
    paketi_gyro = [p for p in paketi if p.id == 1]
    fvz_g, sig_g = sestavi_podatke_cesta(paketi_gyro)

    if len(sig_g) == 0:
        print("Ni Gyro podatkov za cesto!")
        return

    st_vzorcev_g_5s = int(fvz_g * okno_sekunde)
    stevilo_oken = len(sig_g) // st_vzorcev_g_5s
    zbrane_vrstice = []

    for i in range(stevilo_oken):
        zacetek_g = i * st_vzorcev_g_5s
        konec_g = (i + 1) * st_vzorcev_g_5s
        okno_gyro = sig_g[zacetek_g:konec_g]
        
        znac_gyro = izracunaj_znacilke_cesta(okno_gyro, predpona="Gyro")
        

        vrstica = {**znac_gyro}
        vrstica['casovno_okno_sekunde'] = i * okno_sekunde
        zbrane_vrstice.append(vrstica)

    df = pd.DataFrame(zbrane_vrstice)
    df.to_csv(izhodni_csv, index=False)
    print(f"Dataset za Ceste: {izhodni_csv}")


def procesiraj_vse(vhodni_bin):
    print(f"Obdelava: {vhodni_bin}")
    

    paketi = preberi_bin_v_pakete(vhodni_bin)
    if not paketi: return
    

    ime =os.path.splitext(os.path.basename(vhodni_bin))[0]
    csv_policaji = f"inference_policaji_{ime}.csv"
    generiraj_csv_policaji(paketi, csv_policaji)
    

    ime =os.path.splitext(os.path.basename(vhodni_bin))[0]
    csv_cesta = f"inference_cesta_{ime}.csv"
    generiraj_csv_cesta(paketi, csv_cesta)
    return csv_policaji, csv_cesta
