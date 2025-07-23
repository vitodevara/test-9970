# sdn_ai_placement/qos_simulator.py

import pandas as pd
import networkx as nx
import os
import random

class QoSSimulator:
    def __init__(self, qws_data_path=None):
        """
        Inisialisasi simulator QoS.
        Jika qws_data_path diberikan, akan memuat data dari file TXT.
        Jika tidak, akan menggunakan data dummy.
        """
        self.qws_data = self._load_qws_data(qws_data_path)
        
        # Default link properties (digunakan jika tidak ada data QWS spesifik)
        self.default_link_latency = 5 # ms per hop
        self.default_link_bandwidth = 100 # Mbps per link
        self.default_link_loss = 0.01 # 1% loss per hop

        # Hitung batas min/max dari data QWS untuk normalisasi reward
        self._calculate_qos_bounds()

    def _load_qws_data(self, path):
        """
        Memuat data QWS dari file TXT yang dipisahkan koma.
        Mem-parsing kolom: Response Time (0), Throughput (2), Reliability (4).
        """
        data = []
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split(',')
                            if len(parts) >= 11: # Pastikan ada cukup kolom
                                try:
                                    response_time = float(parts[0]) # Kolom (1) Response Time
                                    # Availability = float(parts[1]) # Kolom (2) Availability
                                    throughput = float(parts[2])    # Kolom (3) Throughput
                                    # Successability = float(parts[3])# Kolom (4) Successability
                                    reliability = float(parts[4])   # Kolom (5) Reliability
                                    # Compliance = float(parts[5])
                                    # Best Practices = float(parts[6])
                                    latency = float(parts[7])       # Kolom (8) Latency
                                    # Documentation = float(parts[8])
                                    service_name = parts[9].strip() # Kolom (10) Service Name
                                    # wsdl_address = parts[10].strip()

                                    # Gunakan Latency dari kolom 8 jika tersedia, jika tidak, gunakan Response Time
                                    final_latency = latency if latency is not None else response_time

                                    data.append({
                                        'Service': service_name,
                                        'Latency': final_latency,
                                        'Throughput': throughput,
                                        'Reliability': reliability / 100.0 # Normalisasi ke 0-1
                                    })
                                except ValueError as ve:
                                    print(f"Melewatkan baris karena kesalahan konversi data: {line} - {ve}")
                                except IndexError as ie:
                                    print(f"Melewatkan baris karena format tidak lengkap: {line} - {ie}")
                if data:
                    df = pd.DataFrame(data).set_index('Service')
                    print(f"Berhasil memuat data QWS dari {path}. Jumlah entri: {len(df)}")
                    return df
                else:
                    print(f"Tidak ada data valid yang ditemukan di {path}. Menggunakan data dummy.")
                    return self._create_dummy_qws_data()

            except Exception as e:
                print(f"Gagal memuat data QWS dari {path}: {e}. Menggunakan data dummy.")
                return self._create_dummy_qws_data()
        else:
            print("Path data QWS tidak valid atau tidak ada. Menggunakan data dummy.")
            return self._create_dummy_qws_data()

    def _create_dummy_qws_data(self):
        """Membuat data QWS dummy untuk demonstrasi."""
        data = {
            'Service': ['VoiceCall', 'VideoStream', 'WebBrowsing', 'FileTransfer'],
            'Latency': [50.0, 100.0, 30.0, 200.0], # ms
            'Throughput': [10.0, 50.0, 80.0, 20.0], # Mbps
            'Reliability': [0.99, 0.95, 0.999, 0.90] # 0-1
        }
        return pd.DataFrame(data).set_index('Service')

    def _calculate_qos_bounds(self):
        """
        Menghitung batas atas dan bawah untuk metrik QoS dari data QWS
        untuk membantu normalisasi reward.
        """
        if not self.qws_data.empty:
            self.min_qws_latency = self.qws_data['Latency'].min()
            self.max_qws_latency = self.qws_data['Latency'].max()
            self.min_qws_throughput = self.qws_data['Throughput'].min()
            self.max_qws_throughput = self.qws_data['Throughput'].max()
            self.min_qws_reliability = self.qws_data['Reliability'].min()
            self.max_qws_reliability = self.qws_data['Reliability'].max()
        else:
            # Fallback ke nilai default jika data QWS kosong
            self.min_qws_latency = 1.0
            self.max_qws_latency = 1000.0
            self.min_qws_throughput = 1.0
            self.max_qws_throughput = 100.0
            self.min_qws_reliability = 0.5
            self.max_qws_reliability = 1.0


    def calculate_qos(self, graph: nx.Graph, controller_nodes: list, service_type='WebBrowsing'):
        """
        Menghitung metrik QoS (latensi, throughput, keandalan) untuk topologi
        dengan penempatan controller yang diberikan.
        Ini adalah simulasi, bukan pengukuran real-time.

        Args:
            graph (nx.Graph): Objek NetworkX yang merepresentasikan topologi.
            controller_nodes (list): Daftar node ID tempat controller ditempatkan.
            service_type (str): Jenis layanan untuk mengambil faktor QoS dari QWS.

        Returns:
            dict: Kamus berisi 'latency', 'throughput', 'reliability'.
        """
        if not controller_nodes:
            # Jika tidak ada controller, QoS sangat buruk
            return {'latency': float('inf'), 'throughput': 0, 'reliability': 0}

        # Ambil rata-rata metrik QWS dari dataset untuk jenis layanan yang diminta
        # Jika service_type tidak ada, gunakan rata-rata dari seluruh dataset QWS
        if service_type in self.qws_data.index:
            qos_base = self.qws_data.loc[service_type]
        else:
            # Jika service_type tidak ditemukan, gunakan rata-rata dari seluruh dataset
            if not self.qws_data.empty:
                qos_base = self.qws_data.mean()
                print(f"Service type '{service_type}' tidak ditemukan. Menggunakan rata-rata metrik QWS.")
            else:
                # Jika data QWS juga kosong, gunakan nilai default link
                qos_base = pd.Series({
                    'Latency': self.default_link_latency,
                    'Throughput': self.default_link_bandwidth,
                    'Reliability': 1 - self.default_link_loss
                })
                print("Data QWS kosong. Menggunakan nilai default link.")

        # Asumsi: Latensi per hop dipengaruhi oleh Latency QWS,
        # Throughput per link dibatasi oleh Throughput QWS,
        # Loss per hop dipengaruhi oleh Reliability QWS.
        
        # Gunakan nilai dari QWS sebagai dasar, dan skala dengan jumlah hop
        # Ini adalah model simulasi sederhana.
        
        total_path_latency = 0
        min_path_throughput = float('inf')
        total_path_loss_prob = 0 # Probabilitas kumulatif kehilangan paket

        reachable_nodes_count = 0

        for node in graph.nodes():
            if node not in controller_nodes: # Hanya hitung untuk node yang bukan controller
                min_dist = float('inf')
                
                # Temukan controller terdekat untuk node ini
                for c_node in controller_nodes:
                    try:
                        path_length = nx.shortest_path_length(graph, source=node, target=c_node)
                        if path_length < min_dist:
                            min_dist = path_length
                    except nx.NetworkXNoPath:
                        continue # Lewati node yang tidak bisa dijangkau

                if min_dist != float('inf'): # Jika node dapat dijangkau
                    reachable_nodes_count += 1
                    
                    # Latensi: Latency QWS dikalikan dengan jumlah hop
                    # Ini adalah penyederhanaan. Dalam skenario nyata, mungkin ada latensi per link.
                    path_latency = qos_base['Latency'] * min_dist
                    total_path_latency += path_latency

                    # Throughput: Throughput QWS sebagai batas atas, bisa dibatasi oleh link terlemah
                    # Untuk simulasi sederhana, kita ambil min_path_throughput dari semua jalur
                    path_throughput = qos_base['Throughput']
                    min_path_throughput = min(min_path_throughput, path_throughput)

                    # Keandalan: Probabilitas kehilangan paket kumulatif
                    # P(no loss per hop) = Reliability QWS
                    # P(no loss total) = (Reliability QWS) ^ num_hops
                    prob_no_loss_per_hop = qos_base['Reliability']
                    prob_no_loss_total = prob_no_loss_per_hop ** min_dist
                    total_path_loss_prob += (1 - prob_no_loss_total)
                # else: node tidak dapat dijangkau, akan menyebabkan QoS sangat buruk
                # ini sudah ditangani oleh inisialisasi awal float('inf') dan 0

        if reachable_nodes_count > 0:
            avg_latency = total_path_latency / reachable_nodes_count
            avg_reliability = 1 - (total_path_loss_prob / reachable_nodes_count)
        else:
            # Jika tidak ada node yang dapat dijangkau (atau semua node adalah controller),
            # atau topologi tidak terhubung ke controller
            avg_latency = float('inf')
            avg_reliability = 0

        # Throughput adalah throughput minimum yang ditemukan di seluruh jalur
        final_throughput = min_path_throughput if min_path_throughput != float('inf') else 0

        return {
            'latency': avg_latency,
            'throughput': final_throughput,
            'reliability': avg_reliability
        }

# Contoh penggunaan (tidak perlu dijalankan sebagai skrip utama)
if __name__ == '__main__':
    # Buat topologi dummy untuk pengujian
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]) # Ring topology

    # Buat file dummy qws2.txt untuk pengujian
    dummy_qws_content = """
    ##########################################################################
    ## QWS Dataset 2.0
    ##########################################################################
    ## Attributes or Features
    ## Format: (1) Response Time
    ## Format: (2) Availability
    ## Format: (3) Throughput
    ## Format: (4) Successability
    ## Format: (5) Reliability
    ## Format: (6) Compliance
    ## Format: (7) Best Practices
    ## Format: (8) Latency
    ## Format: (9) Documentation
    ## Format: (10) Service Name
    ## Format: (11) WSDL Address
    ##########################################################################
    302.75,89,7.1,90,73,78,80,187.75,32,MAPPMatching,http://xml.assessment.com/service/MAPPMatching.asmx?wsdl
    482,85,16,95,73,100,84,1,2,Compound2,http://www.mssoapinterop.org/asmx/WSDL/compound2.wsdl
    126.17,98,12,100,67,78,82,22.77,89,GBNIRHolidayDates,http://www.holidaywebservice.com/Holidays/GBNIR/Dates/GBNIRHolidayDates.asmx?WSDL
    107,87,1.9,95,73,89,62,58.33,93,CasUsers,http://galex.stsci.edu/casjobs/CasUsers.asmx?WSDL
    """
    with open('qws2.txt', 'w') as f:
        f.write(dummy_qws_content)

    qos_sim = QoSSimulator(qws_data_path='qws2.txt')

    # Uji penempatan controller yang berbeda
    print("--- Simulasi QoS dengan 1 controller di node 1 (WebBrowsing) ---")
    qos_1_controller = qos_sim.calculate_qos(G, [1], 'WebBrowsing') # 'WebBrowsing' akan menggunakan rata-rata karena tidak ada di dummy
    print(f"QoS: {qos_1_controller}")

    print("\n--- Simulasi QoS dengan 2 controller di node 1 dan 3 (Compound2) ---")
    qos_2_controllers = qos_sim.calculate_qos(G, [1, 3], 'Compound2')
    print(f"QoS: {qos_2_controllers}")

    print("\n--- Simulasi QoS dengan 0 controller ---")
    qos_0_controllers = qos_sim.calculate_qos(G, [], 'MAPPMatching')
    print(f"QoS: {qos_0_controllers}")

    # Bersihkan file dummy
    os.remove('qws2.txt')