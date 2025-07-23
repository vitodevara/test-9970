# sdn_ai_placement/genetic_algorithm.py

import random
import networkx as nx
import numpy as np
from deap import base, creator, tools, algorithms
from qos_simulator import QoSSimulator

class GeneticAlgorithmPlacement:
    def __init__(self, graph: nx.Graph, num_controllers: int, qos_sim: QoSSimulator):
        """
        Inisialisasi Algoritma Genetika untuk penempatan controller.

        Args:
            graph (nx.Graph): Topologi jaringan.
            num_controllers (int): Jumlah controller yang akan ditempatkan.
            qos_sim (QoSSimulator): Instance dari simulator QoS.
        """
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.num_controllers = num_controllers
        self.qos_sim = qos_sim

        # DEAP setup
        # Definisikan tipe fitness: kita ingin memaksimalkan (bobot 1.0) QoS score
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # Definisikan tipe individu: list dari integer (node IDs) dengan fitness
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Atribut generator: memilih node ID secara acak
        # Node ID diasumsikan 0 hingga num_nodes-1
        self.toolbox.register("attr_node", random.randrange, self.num_nodes)

        # Inisialisasi individu: daftar unik dari num_controllers node ID
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_node, self.num_controllers)
        # Pastikan node unik dalam individu
        self.toolbox.decorate("individual", self._ensure_unique_nodes)

        # Inisialisasi populasi
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Operator evaluasi (fitness function)
        self.toolbox.register("evaluate", self._evaluate_placement)

        # Operator seleksi: Tournament Selection
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # Operator crossover: Two-point crossover
        self.toolbox.register("mate", tools.cxTwoPoint)

        # Operator mutasi: Mutasi acak node ID
        # Mutasi akan mengubah satu atau lebih gen (node ID) dalam individu.
        # Penting: Setelah mutasi, kita perlu memastikan keunikan lagi.
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.num_nodes - 1, indpb=0.1)
        # Pastikan mutasi juga menghasilkan node unik
        self.toolbox.decorate("mutate", self._ensure_unique_nodes)

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

    def _ensure_unique_nodes(self, individual):
        """
        Decorator untuk memastikan semua node dalam individu adalah unik.
        Jika ada duplikat, ganti dengan node acak yang belum ada.
        Ini penting setelah crossover dan mutasi.
        """
        # Ubah ke set untuk mendapatkan elemen unik
        unique_nodes = list(set(individual))
        
        # Jika jumlah node unik kurang dari yang seharusnya, tambahkan node acak yang belum ada
        while len(unique_nodes) < self.num_controllers:
            new_node = random.randrange(self.num_nodes)
            if new_node not in unique_nodes:
                unique_nodes.append(new_node)
        
        # Jika ada lebih banyak node unik dari yang dibutuhkan, potong
        if len(unique_nodes) > self.num_controllers:
            unique_nodes = random.sample(unique_nodes, self.num_controllers)

        # Pastikan individu adalah tipe creator.Individual
        individual[:] = unique_nodes # Perbarui individu di tempat
        return individual

    def _evaluate_placement(self, individual):
        """
        Fungsi kebugaran: Menghitung QoS dari penempatan controller yang diberikan.
        Mengembalikan tuple (qos_score,) karena DEAP mengharapkan tuple untuk fitness.
        """
        # Pastikan individu memiliki jumlah controller yang benar dan unik
        # _ensure_unique_nodes seharusnya sudah menangani ini, tapi sebagai fallback
        controller_nodes = list(set(individual))
        if len(controller_nodes) != self.num_controllers:
            # Ini adalah kondisi yang tidak seharusnya terjadi jika _ensure_unique_nodes bekerja
            # Berikan fitness yang sangat rendah sebagai penalti
            return (0.0,)

        # Hitung QoS menggunakan simulator kita
        qos_metrics = self.qos_sim.calculate_qos(self.graph, controller_nodes)

        # Fungsi reward/fitness yang sama dengan DRL untuk konsistensi
        # Latensi yang lebih rendah, throughput lebih tinggi, reliability lebih tinggi lebih baik.
        
        # Normalisasi Latency (semakin rendah semakin baik)
        latency_component = 0
        if qos_metrics['latency'] != float('inf') and (self.max_qws_latency - self.min_qws_latency) > 0:
            latency_component = (self.max_qws_latency - qos_metrics['latency']) / (self.max_qws_latency - self.min_qws_latency)
            latency_component = max(0, min(1, latency_component))
        elif qos_metrics['latency'] == float('inf'):
            latency_component = 0
        else:
            latency_component = 0.5

        # Normalisasi Throughput (semakin tinggi semakin baik)
        throughput_component = 0
        if (self.max_qws_throughput - self.min_qws_throughput) > 0:
            throughput_component = (qos_metrics['throughput'] - self.min_qws_throughput) / (self.max_qws_throughput - self.min_qws_throughput)
            throughput_component = max(0, min(1, throughput_component))
        else:
            throughput_component = 0.5

        # Normalisasi Reliability (semakin tinggi semakin baik)
        reliability_component = 0
        if (self.max_qws_reliability - self.min_qws_reliability) > 0:
            reliability_component = (qos_metrics['reliability'] - self.min_qws_reliability) / (self.max_qws_reliability - self.min_qws_reliability)
            reliability_component = max(0, min(1, reliability_component))
        else:
            reliability_component = 0.5

        # Gabungkan komponen-komponen ini. Bobot yang sama dengan DRL.
        qos_score = (0.4 * latency_component + 0.3 * throughput_component + 0.3 * reliability_component) * 100

        # Penalti jika ada node yang tidak bisa dijangkau controller
        if qos_metrics['latency'] == float('inf') or qos_metrics['throughput'] == 0 or qos_metrics['reliability'] == 0:
             qos_score -= 200 # Penalti besar jika penempatan buruk

        return (qos_score,) # DEAP mengharapkan tuple

    def run(self, population_size=50, generations=100, cxpb=0.7, mutpb=0.2):
        """
        Menjalankan Algoritma Genetika.

        Args:
            population_size (int): Ukuran populasi.
            generations (int): Jumlah generasi untuk evolusi.
            cxpb (float): Probabilitas crossover.
            mutpb (float): Probabilitas mutasi.

        Returns:
            tuple: (best_individual, best_fitness)
        """
        pop = self.toolbox.population(n=population_size)
        
        # Evaluasi populasi awal
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Objek statistik untuk melacak evolusi
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Objek Hall of Fame untuk menyimpan individu terbaik yang pernah ditemukan
        hof = tools.HallOfFame(1) # Hanya menyimpan 1 individu terbaik

        # Jalankan algoritma genetika utama
        # algorithms.eaSimple adalah algoritma GA dasar yang paling umum
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb, mutpb, generations,
                                       stats=stats, halloffame=hof, verbose=False)

        best_individual = hof[0]
        best_fitness = hof[0].fitness.values[0]

        print(f"GA selesai. Individu terbaik: {best_individual}, Fitness terbaik: {best_fitness:.2f}")
        return best_individual, best_fitness

# Contoh penggunaan (tidak perlu dijalankan sebagai skrip utama)
if __name__ == '__main__':
    from qos_simulator import QoSSimulator
    from topology_loader import load_topology_from_gml, get_topology_zoo_path
    import os

    # Siapkan simulator QoS
    dummy_qws_content = """
    302.75,89,7.1,90,73,78,80,187.75,32,MAPPMatching,http://xml.assessment.com/service/MAPPMatching.asmx?wsdl
    482,85,16,95,73,100,84,1,2,Compound2,http://www.mssoapinterop.org/asmx/WSDL/compound2.wsdl
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
        
        ga_solver = GeneticAlgorithmPlacement(G_test, num_controllers_to_place, qos_sim)
        
        print("Memulai Algoritma Genetika...")
        best_placement, best_score = ga_solver.run(population_size=50, generations=50)

        print(f"\nPenempatan controller terbaik dari GA: {best_placement}")
        qos_final = qos_sim.calculate_qos(G_test, best_placement)
        print(f"QoS dari penempatan GA: {qos_final}")
    else:
        print("Gagal menginisialisasi GA karena topologi tidak tersedia.")
    
    # Bersihkan file dummy
    os.remove('qws2.txt')