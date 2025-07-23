# sdn_ai_placement/main_simulation.py

import os
import networkx as nx
from datetime import datetime
import time
import numpy as np # Diperlukan untuk np.reshape di DRL

# Impor semua modul yang diperlukan
from topology_loader import load_topology_from_gml, get_topology_zoo_path
from qos_simulator import QoSSimulator
from traditional_placement import find_central_nodes
from drl_environment import SDNPlacementEnv
from drl_agent_dqn import DQNAgent
from genetic_algorithm import GeneticAlgorithmPlacement

def run_simulation(topology_name: str, qws_data_file: str, num_controllers: int,
                   drl_episodes: int = 200, ga_generations: int = 100,
                   ga_population_size: int = 50):
    """
    Menjalankan simulasi penempatan controller untuk topologi dan jumlah controller tertentu.

    Args:
        topology_name (str): Nama file topologi GML (misal: 'GtsCe.graphml').
        qws_data_file (str): Nama file QWS data TXT (misal: 'qws2.txt').
        num_controllers (int): Jumlah controller yang akan ditempatkan.
        drl_episodes (int): Jumlah episode pelatihan untuk DRL.
        ga_generations (int): Jumlah generasi untuk GA.
        ga_population_size (int): Ukuran populasi untuk GA.
    """
    print(f"\n--- Memulai Simulasi untuk Topologi: {topology_name}, Jumlah Controller: {num_controllers} ---")

    # 1. Muat Topologi
    topology_path = os.path.join(get_topology_zoo_path(), topology_name)
    graph = load_topology_from_gml(topology_path)

    if graph is None:
        print(f"Melewatkan simulasi untuk {topology_name} karena gagal memuat topologi.")
        return

    num_nodes = graph.number_of_nodes()
    if num_controllers > num_nodes:
        print(f"Jumlah controller ({num_controllers}) lebih besar dari jumlah node ({num_nodes}) di topologi {topology_name}. Menyesuaikan jumlah controller menjadi {num_nodes}.")
        num_controllers = num_nodes
    if num_controllers == 0:
        print("Jumlah controller adalah 0, tidak ada yang bisa ditempatkan. Melewatkan simulasi.")
        return
    
    # 2. Inisialisasi Simulator QoS
    qws_data_path_full = os.path.join(os.path.dirname(os.path.abspath(__file__)), qws_data_file)
    qos_sim = QoSSimulator(qws_data_path=qws_data_path_full)

    results = {}

    # --- Metode Tradisional ---
    print("\n--- Menjalankan Metode Tradisional ---")
    start_time = time.time()
    traditional_placement = find_central_nodes(graph, num_controllers)
    traditional_qos = qos_sim.calculate_qos(graph, traditional_placement)
    end_time = time.time()
    traditional_time = end_time - start_time

    results['Traditional'] = {
        'placement': traditional_placement,
        'qos': traditional_qos,
        'time': traditional_time
    }
    print(f"Penempatan Tradisional: {traditional_placement}")
    print(f"QoS Tradisional: Latency={traditional_qos['latency']:.2f}ms, Throughput={traditional_qos['throughput']:.2f}Mbps, Reliability={traditional_qos['reliability']:.4f}")
    print(f"Waktu Eksekusi Tradisional: {traditional_time:.4f} detik")

    # --- Metode Deep Reinforcement Learning (DRL) ---
    print("\n--- Menjalankan Metode Deep Reinforcement Learning (DQN) ---")
    start_time = time.time()
    env = SDNPlacementEnv(graph, num_controllers, qos_sim)
    state_size = env.get_state_space_size()
    action_size = env.get_action_space_size()
    agent = DQNAgent(state_size, action_size)

    # Pelatihan DRL
    print(f"Memulai pelatihan DQN untuk {drl_episodes} episode...")
    for e in range(drl_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        
        # Loop untuk setiap langkah dalam episode (menempatkan num_controllers)
        for step_count in range(num_controllers):
            action = agent.act(state[0]) # Agen memilih aksi
            
            # Tambahan untuk memastikan agen memilih node yang belum ditempati
            # Ini membantu agen belajar lebih cepat di awal, meskipun reward function
            # sudah memberikan penalti untuk duplikasi.
            if state[0][action] == 1: # Jika node sudah memiliki controller
                available_nodes = [i for i, val in enumerate(state[0]) if val == 0]
                if available_nodes:
                    action = random.choice(available_nodes)
                else: # Semua node sudah dipilih, tidak ada aksi valid lagi
                    break 

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            agent.remember(state[0], action, reward, next_state[0], done)
            state = next_state
            total_reward += reward

            if done: # Jika semua controller sudah ditempatkan
                break
        
        # Latih agen setelah setiap episode
        agent.replay()
        # Perbarui target model secara berkala
        if e % 10 == 0: # Setiap 10 episode
            agent.update_target_model()
        
        if (e + 1) % (drl_episodes // 10) == 0 or e == drl_episodes - 1: # Cetak progres
            print(f"  Episode: {e+1}/{drl_episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Dapatkan penempatan terbaik dari agen yang sudah dilatih (eksploitasi penuh)
    agent.epsilon = 0.0 # Matikan eksplorasi untuk evaluasi
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    
    drl_placement = []
    # Loop untuk memilih num_controllers node unik
    while len(drl_placement) < num_controllers:
        action = agent.act(state[0])
        if action not in drl_placement:
            drl_placement.append(action)
        # Update state untuk mempengaruhi pilihan selanjutnya jika agen menggunakan state
        state[0][action] = 1 

    drl_qos = qos_sim.calculate_qos(graph, drl_placement)
    end_time = time.time()
    drl_time = end_time - start_time

    results['DRL'] = {
        'placement': drl_placement,
        'qos': drl_qos,
        'time': drl_time
    }
    print(f"Penempatan DRL: {drl_placement}")
    print(f"QoS DRL: Latency={drl_qos['latency']:.2f}ms, Throughput={drl_qos['throughput']:.2f}Mbps, Reliability={drl_qos['reliability']:.4f}")
    print(f"Waktu Eksekusi DRL: {drl_time:.4f} detik")

    # --- Metode Genetic Algorithm (GA) ---
    print("\n--- Menjalankan Metode Genetic Algorithm (GA) ---")
    start_time = time.time()
    ga_solver = GeneticAlgorithmPlacement(graph, num_controllers, qos_sim)
    best_ga_placement, best_ga_score = ga_solver.run(
        population_size=ga_population_size, generations=ga_generations
    )
    ga_qos = qos_sim.calculate_qos(graph, best_ga_placement)
    end_time = time.time()
    ga_time = end_time - start_time

    results['GA'] = {
        'placement': best_ga_placement,
        'qos': ga_qos,
        'time': ga_time
    }
    print(f"Penempatan GA: {best_ga_placement}")
    print(f"QoS GA: Latency={ga_qos['latency']:.2f}ms, Throughput={ga_qos['throughput']:.2f}Mbps, Reliability={ga_qos['reliability']:.4f}")
    print(f"Waktu Eksekusi GA: {ga_time:.4f} detik")

    # --- Ringkasan Hasil ---
    print("\n--- Ringkasan Hasil Simulasi ---")
    print(f"Topologi: {topology_name}, Jumlah Controller: {num_controllers}")
    print("-" * 50)
    print(f"{'Metode':<15} | {'Penempatan':<20} | {'Latensi (ms)':<15} | {'Throughput (Mbps)':<18} | {'Keandalan':<12} | {'Waktu (s)':<10}")
    print("-" * 100)
    for method, data in results.items():
        placement_str = str(data['placement'])
        if len(placement_str) > 18:
            placement_str = placement_str[:15] + "..."
        print(f"{method:<15} | {placement_str:<20} | {data['qos']['latency']:.2f}{'':<13} | {data['qos']['throughput']:.2f}{'':<16} | {data['qos']['reliability']:.4f}{'':<10} | {data['time']:.4f}")
    print("-" * 100)

if __name__ == '__main__':
    # Konfigurasi simulasi
    # Pastikan file 'qws2.txt' dan 'GtsCe.graphml' ada di direktori yang sama
    # dengan script ini.
    
    simulations_to_run = [
        {'topology': 'GtsCe.graphml', 'qws_data': 'qws2.txt', 'controllers': 2},
        {'topology': 'GtsCe.graphml', 'qws_data': 'qws2.txt', 'controllers': 3},
        # Anda bisa menambahkan topologi lain dari GitHub sk2/topologyzoo
        # Misalnya:
        # {'topology': 'Abilene.graphml', 'qws_data': 'qws2.txt', 'controllers': 2},
        # {'topology': 'GEANT2.graphml', 'qws_data': 'qws2.txt', 'controllers': 3},
    ]

    for sim_config in simulations_to_run:
        run_simulation(
            topology_name=sim_config['topology'],
            qws_data_file=sim_config['qws_data'],
            num_controllers=sim_config['controllers'],
            drl_episodes=200, # Anda bisa meningkatkan ini untuk hasil DRL yang lebih baik
            ga_generations=100, # Anda bisa meningkatkan ini untuk hasil GA yang lebih baik
            ga_population_size=50
        )
    
    print("\nSimulasi Selesai!")
