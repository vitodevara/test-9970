# sdn_ai_placement/drl_environment.py

import networkx as nx
import numpy as np
import random
from qos_simulator import QoSSimulator # Impor simulator QoS kita

class SDNPlacementEnv:
    def __init__(self, graph: nx.Graph, num_controllers: int, qos_sim: QoSSimulator):
        """
        Inisialisasi lingkungan penempatan controller SDN untuk DRL.

        Args:
            graph (nx.Graph): Topologi jaringan.
            num_controllers (int): Jumlah controller yang akan ditempatkan.
            qos_sim (QoSSimulator): Instance dari simulator QoS.
        """
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.num_controllers = num_controllers
        self.qos_sim = qos_sim

        # State: Representasi biner dari node yang dipilih sebagai controller.
        # Ukuran state space adalah num_nodes.
        self.state = np.zeros(self.num_nodes, dtype=int) # 0: bukan controller, 1: controller

        # Action: Memilih node untuk menempatkan controller.
        # Ukuran action space adalah num_nodes (memilih node ke-i).
        self.action_space_size = self.num_nodes

        self.current_placement_step = 0 # Melacak berapa banyak controller yang sudah ditempatkan

        # Ambil batas normalisasi dari qos_sim
        self.min_qws_latency = self.qos_sim.min_qws_latency
        self.max_qws_latency = self.qos_sim.max_qws_latency
        self.min_qws_throughput = self.qos_sim.min_qws_throughput
        self.max_qws_throughput = self.qos_sim.max_qws_throughput
        self.min_qws_reliability = self.qos_sim.min_qws_reliability
        self.max_qws_reliability = self.qos_sim.max_qws_reliability

        # Tambahkan nilai aman untuk menghindari pembagian dengan nol atau rentang nol
        if self.max_qws_latency == self.min_qws_latency:
            self.max_qws_latency = self.min_qws_latency + 1.0
        if self.max_qws_throughput == self.min_qws_throughput:
            self.max_qws_throughput = self.min_qws_throughput + 1.0
        if self.max_qws_reliability == self.min_qws_reliability:
            self.max_qws_reliability = self.min_qws_reliability + 0.01


    def reset(self):
        """
        Mengatur ulang lingkungan ke keadaan awal.
        """
        self.state = np.zeros(self.num_nodes, dtype=int)
        self.current_placement_step = 0
        return self.state

    def step(self, action: int):
        """
        Melakukan satu langkah di lingkungan berdasarkan aksi yang diberikan.

        Args:
            action (int): Node ID (indeks) tempat controller akan ditempatkan.

        Returns:
            tuple: (next_state, reward, done, info)
                next_state (np.array): Keadaan setelah aksi dilakukan.
                reward (float): Hadiah yang diterima setelah aksi.
                done (bool): True jika episode selesai.
                info (dict): Informasi tambahan (opsional).
        """
        # Pastikan aksi valid
        if not (0 <= action < self.num_nodes):
            # Aksi tidak valid, berikan reward negatif besar dan akhiri episode
            return self.state, -100, True, {"message": "Invalid action: out of bounds"}

        # Lakukan aksi: tempatkan controller di node yang dipilih
        # Jika node sudah memiliki controller, ini akan diatasi oleh reward function
        # dengan memberikan reward yang tidak optimal, mendorong agen untuk belajar.
        self.state[action] = 1
        self.current_placement_step += 1

        done = False
        reward = 0
        info = {}

        # Jika sudah menempatkan jumlah controller yang diinginkan, episode selesai
        if self.current_placement_step >= self.num_controllers:
            done = True
            # Hitung reward berdasarkan QoS dari penempatan akhir
            current_controller_nodes = [i for i, val in enumerate(self.state) if val == 1]
            qos_metrics = self.qos_sim.calculate_qos(self.graph, current_controller_nodes)

            # Fungsi reward: Gabungkan latensi, throughput, dan keandalan.
            # Reward harus lebih tinggi untuk latensi rendah, throughput tinggi, reliability tinggi.
            
            # Normalisasi Latency (semakin rendah semakin baik)
            # Skala terbalik: (Max - Actual) / (Max - Min)
            latency_component = 0
            if qos_metrics['latency'] != float('inf') and (self.max_qws_latency - self.min_qws_latency) > 0:
                latency_component = (self.max_qws_latency - qos_metrics['latency']) / (self.max_qws_latency - self.min_qws_latency)
                latency_component = max(0, min(1, latency_component)) # Batasi antara 0 dan 1
            elif qos_metrics['latency'] == float('inf'): # Latensi tak terbatas berarti sangat buruk
                latency_component = 0
            else: # Rentang nol atau kondisi lain yang aneh
                latency_component = 0.5 # Netral

            # Normalisasi Throughput (semakin tinggi semakin baik)
            # Skala langsung: (Actual - Min) / (Max - Min)
            throughput_component = 0
            if (self.max_qws_throughput - self.min_qws_throughput) > 0:
                throughput_component = (qos_metrics['throughput'] - self.min_qws_throughput) / (self.max_qws_throughput - self.min_qws_throughput)
                throughput_component = max(0, min(1, throughput_component)) # Batasi antara 0 dan 1
            else:
                throughput_component = 0.5 # Netral

            # Normalisasi Reliability (semakin tinggi semakin baik)
            # Skala langsung: (Actual - Min) / (Max - Min)
            reliability_component = 0
            if (self.max_qws_reliability - self.min_qws_reliability) > 0:
                reliability_component = (qos_metrics['reliability'] - self.min_qws_reliability) / (self.max_qws_reliability - self.min_qws_reliability)
                reliability_component = max(0, min(1, reliability_component)) # Batasi antara 0 dan 1
            else:
                reliability_component = 0.5 # Netral

            # Gabungkan komponen-komponen ini. Anda bisa menyesuaikan bobotnya.
            # Reward diskalakan agar lebih besar dan lebih mudah dibedakan oleh agen.
            reward = (0.4 * latency_component + 0.3 * throughput_component + 0.3 * reliability_component) * 100

            # Penalti jika ada node yang tidak bisa dijangkau controller
            # Ini akan memberikan reward negatif yang signifikan jika penempatan sangat buruk
            if qos_metrics['latency'] == float('inf') or qos_metrics['throughput'] == 0 or qos_metrics['reliability'] == 0:
                 reward -= 200 # Penalti besar jika penempatan buruk

            # Penalti jika agen memilih node yang sudah ditempati (tidak efisien)
            if len(set(current_controller_nodes)) < self.num_controllers:
                reward -= 50 # Penalti karena duplikasi penempatan
                info["message"] = "Duplicate node selected"

        return self.state, reward, done, info

    def get_state_space_size(self):
        return self.num_nodes

    def get_action_space_size(self):
        return self.action_space_size

# Contoh penggunaan (tidak perlu dijalankan sebagai skrip utama)
if __name__ == '__main__':
    from qos_simulator import QoSSimulator
    from topology_loader import load_topology_from_gml, get_topology_zoo_path
    import os

    # Siapkan simulator QoS
    # Buat file dummy qws2.txt untuk pengujian
    dummy_qws_content = """
    302.75,89,7.1,90,73,78,80,187.75,32,MAPPMatching,http://xml.assessment.com/service/MAPPMatching.asmx?wsdl
    482,85,16,95,73,100,84,1,2,Compound2,http://www.mssoapinterop.org/asmx/WSDL/compound2.wsdl
    126.17,98,12,100,67,78,82,22.77,89,GBNIRHolidayDates,http://www.holidaywebservice.com/Holidays/GBNIR/Dates/GBNIRHolidayDates.asmx?WSDL
    107,87,1.9,95,73,89,62,58.33,93,CasUsers,http://galex.stsci.edu/casjobs/CasUsers.asmx?WSDL
    """
    with open('qws2.txt', 'w') as f:
        f.write(dummy_qws_content)
    qos_sim = QoSSimulator(qws_data_path='qws2.txt')

    # Muat topologi contoh
    topology_dir = get_topology_zoo_path()
    sample_gml_file = os.path.join(topology_dir, "GtsCe.graphml")
    
    if not os.path.exists(sample_gml_file):
        print(f"File {sample_gml_file} tidak ditemukan. Membuat topologi dummy.")
        G_test = nx.Graph()
        G_test.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])
    else:
        G_test = load_topology_from_gml(sample_gml_file)

    if G_test:
        num_controllers_to_place = 2
        env = SDNPlacementEnv(G_test, num_controllers_to_place, qos_sim)

        print(f"Ukuran state space: {env.get_state_space_size()}")
        print(f"Ukuran action space: {env.get_action_space_size()}")

        # Simulasikan beberapa episode
        print("\n--- Simulasi Episode 1 ---")
        state = env.reset()
        print(f"State awal: {state}")
        done = False
        total_reward = 0

        for step_count in range(num_controllers_to_place):
            action = random.randint(0, env.get_action_space_size() - 1)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            print(f"Aksi: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}, Info: {info}")
            if done:
                break

        print(f"Total reward Episode 1: {total_reward}")
        print(f"Penempatan controller akhir: {[i for i, val in enumerate(env.state) if val == 1]}")

        print("\n--- Simulasi Episode 2 (Coba penempatan duplikat) ---")
        state = env.reset()
        print(f"State awal: {state}")
        done = False
        total_reward = 0

        # Tempatkan controller di node 0
        next_state, reward, done, info = env.step(0)
        total_reward += reward
        print(f"Aksi: 0, Next State: {next_state}, Reward: {reward}, Done: {done}, Info: {info}")

        # Coba tempatkan lagi di node 0 (duplikat)
        if not done:
            next_state, reward, done, info = env.step(0)
            total_reward += reward
            print(f"Aksi: 0 (duplikat), Next State: {next_state}, Reward: {reward}, Done: {done}, Info: {info}")
        
        print(f"Total reward Episode 2: {total_reward}")
        print(f"Penempatan controller akhir: {[i for i, val in enumerate(env.state) if val == 1]}")

    else:
        print("Gagal menginisialisasi lingkungan karena topologi tidak tersedia.")
    
    # Bersihkan file dummy
    os.remove('qws2.txt')