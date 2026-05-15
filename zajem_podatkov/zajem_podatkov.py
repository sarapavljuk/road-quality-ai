import serial
import time

SERIAL_PORT = 'COM5'
BAUD_RATE = 115200

def record_sequence():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print("\n" + "=" * 50)
        print(" ZAJEM PODATKOV")
        print("=" * 50)
    except Exception as e:
        print(f"NAPAKA: Ni mogoce odpreti vrat {SERIAL_PORT}.")
        print("Preveri povezavo in zapri ostale programe.")
        return

    file_name = input("Vnesi ime datoteke (brez koncnice): ").strip()
    if not file_name:
        file_name = "posnetek"

    try:
        record_duration = float(input("Vnesi dolzino posnetka v sekundah: ").strip())
        if record_duration <= 0:
            print("Dolzina mora biti vecja od 0.")
            ser.close()
            return
    except ValueError:
        print("Napacen vnos za dolzino posnetka.")
        ser.close()
        return

    file_name = f"{file_name}.BIN"

    print(f"\nDatoteka: {file_name}")
    input("Pritisni ENTER za zacetek snemanja...")

    ser.reset_input_buffer()

    with open(file_name, "wb") as f:
        print(f" >>> SNEMAM ({record_duration}s)... ", end="", flush=True)

        start_time = time.time()
        while (time.time() - start_time) < record_duration:
            if ser.in_waiting > 0:
                raw_data = ser.read(ser.in_waiting)
                f.write(raw_data)

    print("KONCANO!")
    print(f"Podatki so shranjeni v datoteko {file_name}.")

    ser.close()

if __name__ == "__main__":
    record_sequence()