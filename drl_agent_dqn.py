# sdn_ai_placement/drl_agent_dqn.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, learning_rate=0.001,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=32, replay_buffer_size=2000):
        """
        Inisialisasi agen Deep Q-Network (DQN).

        Args:
            state_size (int): Ukuran state space (jumlah node dalam topologi).
            action_size (int): Ukuran action space (jumlah node dalam topologi).
            learning_rate (float): Tingkat pembelajaran untuk optimizer.
            gamma (float): Discount factor untuk reward di masa depan.
            epsilon (float): Probabilitas eksplorasi awal.
            epsilon_min (float): Probabilitas eksplorasi minimum.
            epsilon_decay (float): Faktor peluruhan epsilon per episode.
            batch_size (int): Ukuran batch untuk melatih jaringan.
            replay_buffer_size (int): Ukuran maksimum replay buffer.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=replay_buffer_size) # Replay buffer

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model() # Inisialisasi target model

    def _build_model(self):
        """
        Membangun jaringan saraf untuk Q-Network.
        Model ini akan memetakan state (array biner penempatan controller)
        ke Q-values untuk setiap kemungkinan aksi (penempatan controller di node).
        """
        model = Sequential()
        # Input layer: state_size (jumlah node)
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # Output layer: action_size (jumlah node), satu output per aksi (Q-value)
        model.add(Dense(self.action_size, activation='linear')) # Linear activation untuk Q-values
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """
        Memperbarui bobot target model dari model utama.
        Dilakukan secara berkala untuk stabilitas pelatihan.
        """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Menyimpan transisi ke replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Memilih aksi berdasarkan epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            # Eksplorasi: Pilih aksi acak
            return random.randrange(self.action_size)
        # Eksploitasi: Pilih aksi dengan Q-value tertinggi
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):
        """
        Melatih jaringan dari sampel acak di replay buffer.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        # Ekstrak komponen dari minibatch
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Hitung Q-values target
        # Q(s,a) = r + gamma * max(Q(s',a'))
        target_q_values = self.target_model.predict(next_states, verbose=0)
        target_q = rewards + self.gamma * np.amax(target_q_values, axis=1) * (1 - dones)

        # Dapatkan Q-values saat ini dari model utama
        current_q_values = self.model.predict(states, verbose=0)
        
        # Perbarui Q-value untuk aksi yang diambil
        # Target Q-value hanya berlaku untuk aksi yang diambil
        # Q-values untuk aksi lain tetap sama
        target_f = current_q_values.copy()
        target_f[np.arange(self.batch_size), actions] = target_q

        # Latih model
        self.model.fit(states, target_f, epochs=1, verbose=0)

        # Kurangi epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        """Menyimpan bobot model."""
        self.model.save_weights(name)

    def load(self, name):
        """Memuat bobot model."""
        self.model.load_weights(name)

# Contoh penggunaan (tidak perlu dijalankan sebagai skrip utama)
if __name__ == '__main__':
    from drl_environment import SDNPlacementEnv
    from qos_simulator import QoSSimulator
    from topology_loader import load_topology_from_gml, get_topology_zoo_path
    import os

    # Siapkan simulator QoS
    qos_sim = QoSSimulator()

    # Muat topologi contoh
    topology_zoo_dir = get_topology_zoo_path()
    sample_gml_file = os.path.join(topology_zoo_dir, "Abilene.gml")
    
    if not os.path.exists(sample_gml_file):
        print(f"File {sample_gml_file} tidak ditemukan. Membuat topologi dummy.")
        G_test = nx.Graph()
        G_test.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])
    else:
        G_test = load_topology_from_gml(sample_gml_file)

    if G_test:
        num_controllers_to_place = 2
        env = SDNPlacementEnv(G_test, num_controllers_to_place, qos_sim)

        state_size = env.get_state_space_size()
        action_size = env.get_action_space_size()
        agent = DQNAgent(state_size, action_size)

        episodes = 100 # Jumlah episode pelatihan singkat untuk contoh

        print(f"Memulai pelatihan DQN untuk {episodes} episode...")
        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, state_size]) # Reshape untuk input model
            done = False
            total_reward = 0
            
            # Loop untuk setiap langkah dalam episode
            for step_count in range(num_controllers_to_place): # Maksimal langkah adalah num_controllers
                action = agent.act(state[0]) # Ambil aksi dari agen
                
                # Pastikan aksi valid (node yang belum dipilih)
                # Ini adalah cara sederhana untuk menghindari memilih node yang sama berulang kali
                # Agen yang lebih cerdas akan belajar ini dari reward negatif
                valid_actions = [i for i, val in enumerate(state[0]) if val == 0]
                if not valid_actions: # Jika semua node sudah dipilih, ini adalah kasus yang aneh
                    break
                
                if action not in valid_actions:
                    # Jika agen memilih aksi yang tidak valid, pilih aksi valid secara acak
                    action = random.choice(valid_actions)

                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                
                # Ingat transisi
                agent.remember(state[0], action, reward, next_state[0], done)
                
                state = next_state
                total_reward += reward

                if done:
                    # Episode selesai, latih agen dan perbarui target model
                    agent.replay()
                    if e % 10 == 0: # Perbarui target model setiap 10 episode
                        agent.update_target_model()
                    print(f"Episode: {e+1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                    break
            
            # Jika episode berakhir sebelum mencapai num_controllers (misal karena invalid action)
            if not done and step_count == num_controllers_to_place - 1:
                # Ini adalah kasus di mana kita mencapai batas langkah tanpa 'done'
                # Ini harus ditangani oleh lingkungan, tapi untuk berjaga-jaga
                current_controller_nodes = [i for i, val in enumerate(env.state) if val == 1]
                qos_metrics = qos_sim.calculate_qos(G_test, current_controller_nodes)
                # Hitung reward final seperti di env.step()
                latency_component = (env.max_possible_latency - qos_metrics['latency']) / env.max_possible_latency
                throughput_component = qos_metrics['throughput'] / env.max_possible_throughput
                reliability_component = qos_metrics['reliability'] / env.max_possible_reliability
                final_reward = (0.4 * latency_component + 0.3 * throughput_component + 0.3 * reliability_component) * 100
                if qos_metrics['latency'] == float('inf') or qos_metrics['throughput'] == 0 or qos_metrics['reliability'] == 0:
                     final_reward -= 50
                
                agent.remember(state[0], action, final_reward, next_state[0], True) # Tandai sebagai done
                agent.replay()
                if e % 10 == 0: # Perbarui target model setiap 10 episode
                    agent.update_target_model()
                print(f"Episode: {e+1}/{episodes}, Score: {final_reward:.2f}, Epsilon: {agent.epsilon:.2f} (Early Done)")

        print("\nPelatihan DQN selesai.")
        # Simpan model jika diperlukan
        # agent.save("dqn_controller_placement.h5")

        # Uji agen yang sudah dilatih (eksploitasi penuh)
        print("\n--- Menguji agen yang sudah dilatih ---")
        agent.epsilon = 0.0 # Matikan eksplorasi
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        final_placement = []
        for _ in range(num_controllers_to_place):
            action = agent.act(state[0])
            final_placement.append(action)
            
            # Perbarui state secara manual untuk simulasi
            state[0][action] = 1 
            # Jika kita ingin lebih akurat, panggil env.step() lagi
            # next_state, _, _, _ = env.step(action)
            # state = np.reshape(next_state, [1, state_size])

        print(f"Penempatan controller yang disarankan oleh DQN: {final_placement}")
        qos_final = qos_sim.calculate_qos(G_test, final_placement)
        print(f"QoS dari penempatan DQN: {qos_final}")

    else:
        print("Gagal menginisialisasi agen DQN karena topologi tidak tersedia.")