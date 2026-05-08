import serial
import time

SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
RECORD_DURATION = 5
SAMPLES_PER_CATEGORY = 5

def record_sequence():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print("\n" + "="*50)
        print(" AVTOMATSKI ZAJEMALNIK PODATKOV (20 POSNETKOV)")
        print("="*50)
    except Exception as e:
        print(f"NAPAKA: Ni mogoce odpreti vrat {SERIAL_PORT}.")
        print(f"Preveri povezavo in zapri ostale programe.")
        return

    kategorije = ["policaj", "luknja", "dobra", "slaba"]

    for kat in kategorije:
        print(f"\n\n>>> TRENUTNA KATEGORIJA: {kat.upper()}")
        print("-" * 30)
        
        for i in range(1, SAMPLES_PER_CATEGORY + 1):
            file_name = f"{kat}_{i}.BIN"
            
            print(f"\n[PRIPRAVA] Datoteka: {file_name}")
            input(f"Pritisni ENTER za zacetek snemanja {i}/5...")

            ser.reset_input_buffer()
            
            with open(file_name, "wb") as f:
                print(f"   >>> SNEMAM ({RECORD_DURATION}s)... ", end="", flush=True)
                
                start_time = time.time()
                while (time.time() - start_time) < RECORD_DURATION:
                    if ser.in_waiting > 0:
                        raw_data = ser.read(ser.in_waiting)
                        f.write(raw_data)
                
                print("KONCANO!")
        
        print(f"\n--- Kategorija '{kat}' je v celoti posneta. ---")

    print("\n" + "="*50)
    print("CESTITAMO! Vseh 20 datotek je pripravljenih.")
    print("="*50)
    
    ser.close()

if __name__ == "__main__":
    record_sequence()