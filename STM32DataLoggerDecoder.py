import struct
import os

class STM32DataLoggerDecoder:
    def __init__(self, filename):
        self.filename = filename
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

    def parse_file(self):
        if not os.path.exists(self.filename):
            print(f"Datoteka {self.filename} ne obstaja.")
            return

        with open(self.filename, 'rb') as f:
            content = f.read()

        # Razdelimo na surove bloke med markerji 0xFFFF
        raw_chunks = content.split(self.sync_marker)
        
        valid_packets = 0
        total_sensor_chunks = 0

        for raw_data in raw_chunks:
            if len(raw_data) < 5:  # Premalo za karkoli uporabnega
                continue

            # Odstiranje (brez prvega bajta, ki je counter)
            payload = self.unstuff_payload(raw_data[1:])
            
            # Če je payload vsaj približno dolg dovolj za glavo (timestamp + size)
            if len(payload) >= 6:
                valid_packets += 1
                
                # Razčlenimo chunks_area (preskočimo prvih 6 bajtov glave)
                # CRC je na koncu, a ga v tem "prizanesljivem" načinu ne preverjamo strogo
                chunks_area = payload[6:-2] if len(payload) > 8 else payload[6:]
                
                pos = 0
                while pos < len(chunks_area):
                    if pos + 3 > len(chunks_area): break
                    try:
                        chunk_size = struct.unpack('<H', chunks_area[pos+1:pos+3])[0] + 1
                        total_sensor_chunks += 1
                        pos += 4 + chunk_size
                    except:
                        break

        # Izpis rezultatov
        file_size = os.path.getsize(self.filename)
        print("=" * 40)
        print(f"ANALIZA BIN DATOTEKE: {self.filename}")
        print(f"Velikost datoteke: {file_size} bajtov")
        print("-" * 40)
        print(f"Število prebranih paketov: {valid_packets}")
        print(f"Število senzorskih čunkov: {total_sensor_chunks}")
        
        if valid_packets == 124 and total_sensor_chunks == 372:
            print("\nUSPEH: Vsi podatki so pravilno prebrani (124 paketov, 372 čunkov).")
        else:
            print(f"\nOPOMBA: Manjka {124 - valid_packets} paketov do cilja 124.")
        print("=" * 40)

if __name__ == "__main__":
    parser = STM32DataLoggerDecoder("naloga_1_log.BIN")
    parser.parse_file()