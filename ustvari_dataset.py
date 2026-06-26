import os
import glob
import struct
import numpy as np
import pandas as pd
from dataclasses import dataclass
from STM32DataLoggerDecoder import STM32DataLoggerDecoder

SENSOR_RESOLUTION = {1: 8.75e-3, 2: 6.125e-5, 3: 1.5e-3}

@dataclass
class Paket:
    id: int
    ts: float
    data: np.ndarray

def dekodiraj_datoteko(filename) -> list:
    """Prebere BIN datoteko in vrne zaporedje paketov."""
    dec = STM32DataLoggerDecoder(filename)
    if not os.path.exists(filename):
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
        ts = ts_ms / 1000.0  # pretvorba v sekunde
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
    """Iz paketov sestavi Nx4 matriko: [cas, osX, osY, osZ]"""
    senzor_paketi = [p for p in paketi if p.id == id_senzorja]
    if not senzor_paketi:
        return np.array([])

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

def izracunaj_znacilke(data, prefiks):
    """Iz Nx(3 ali 4) matrike izračuna statistične značilke za Mreže / ML modele."""
    z = {}
    if data is None or len(data) == 0:

        osi = ['x', 'y', 'z']
        if "acc" in prefiks: osi.append('mag')
        for os in osi:
            z[f'{prefiks}_{os}_mean'] = 0.0
            z[f'{prefiks}_{os}_std'] = 0.0
            z[f'{prefiks}_{os}_max'] = 0.0
            z[f'{prefiks}_{os}_ptp'] = 0.0  # peak-to-peak (max - min)
        return z

    osi_imena = ['x', 'y', 'z']
    
    if "acc" in prefiks:
        mag = np.sqrt(data[:, 1]**2 + data[:, 2]**2 + data[:, 3]**2)
        data = np.column_stack((data, mag))
        osi_imena.append('mag')

    for i, os in enumerate(osi_imena):
        stolpec = data[:, i + 1] # +1 ker je na 0-tem indexu čas!
        z[f'{prefiks}_{os}_mean'] = np.mean(stolpec)
        z[f'{prefiks}_{os}_std'] = np.std(stolpec)
        z[f'{prefiks}_{os}_max'] = np.max(stolpec)
        z[f'{prefiks}_{os}_ptp'] = np.ptp(stolpec) # Amplituda (Peak-To-Peak)

    return z

def procesiraj_datoteko(filename, okno_dolzina=5.0, korak=5.0, target_label=None):
    """
    Razreže prebrano datoteko na časovna okna in vrne seznam značilk (features).
    - Za trening: Oknu ni potrebno prekrivanje (korak = 5.0)
    - Za drsečo detekcijo: okno naj drsi naprej (korak = 1.0)
    """
    paketi = dekodiraj_datoteko(filename)
    if not paketi:
        return []

    acc_data = ustvari_zvezni_signal(paketi, 2)
    gyro_data = ustvari_zvezni_signal(paketi, 1)

    if len(acc_data) == 0: return []

    start_time = acc_data[0, 0]
    end_time = acc_data[-1, 0]
    
    okna_features = []
    trenutni_cas = start_time

    while trenutni_cas <= end_time - (okno_dolzina * 0.5): 
        konec_okna = trenutni_cas + okno_dolzina
        
        acc_okno = acc_data[(acc_data[:, 0] >= trenutni_cas) & (acc_data[:, 0] < konec_okna)]
        gyro_okno = gyro_data[(gyro_data[:, 0] >= trenutni_cas) & (gyro_data[:, 0] < konec_okna)] if len(gyro_data) > 0 else []

        zn_acc = izracunaj_znacilke(acc_okno, "acc")
        zn_gyro = izracunaj_znacilke(gyro_okno, "gyro")

        vrstica = {**zn_acc, **zn_gyro}
        
        vrstica['cas_okna_start'] = trenutni_cas - start_time # Relativni čas od 0
        
        if target_label is not None:
            vrstica['target'] = target_label
            
        okna_features.append(vrstica)
        trenutni_cas += korak

    return okna_features

if __name__ == "__main__":
    OUT_DIR = "data_out"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("--- 1. Priprava Učbenega Set-a (Train) ---")
    
    folders = {
        "lezeci": 1,
        "slaba_cesta": 0,
        "dobra_cesta": 0,
        "luknja": 0
    }

    all_train_rows = []
    
    for map_name, label in folders.items():

        bin_files = glob.glob(f"{map_name}/*.BIN")
        if not bin_files:
            bin_files = glob.glob(f"{map_name}/*.bin")

        print(f"Najdeno {len(bin_files)} datotek v mapi '{map_name}'")

        for f in bin_files:
            features = procesiraj_datoteko(f, okno_dolzina=5.0, korak=5.0, target_label=label)
            all_train_rows.extend(features)

    if all_train_rows:
        df_train = pd.DataFrame(all_train_rows)
        train_path = os.path.join(OUT_DIR, "train_dataset.csv")
        df_train.to_csv(train_path, index=False)
        print(f"-> Uspešno shranjeno učno mnozico: {train_path} ({len(df_train)} vrstic)\n")
    else:
        print("-> OPOZORILO: Ni najdenih učnih podatkov! (Preveri če mape obstajajo in imajo .BIN datoteke)\n")


    print("--- 2. Priprava testne vožnje za drsečo detekcijo ---")
    dolgi_kandidati = glob.glob("dolgi*.BIN") 
    
    for dolgi in dolgi_kandidati:
        print(f"Obdelujem dolgo testno vožnjo: {dolgi}")
        features_test = procesiraj_datoteko(dolgi, okno_dolzina=5.0, korak=1.0, target_label=None)
        
        if features_test:
            df_test = pd.DataFrame(features_test)
            test_path = os.path.join(OUT_DIR, f"inference_{os.path.basename(dolgi)}.csv")
            df_test.to_csv(test_path, index=False)
            print(f"-> Uspešno shranjeno drsečo testno množico: {test_path} ({len(df_test)} vrstic)\n")

    print("\nPre-processing je ZAKLJUČEN. Podatki so sedaj pripravljeni za XGBoost!")

