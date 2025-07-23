# sdn_ai_placement/topology_loader.py

import networkx as nx
import os

def load_topology_from_gml(gml_file_path: str) -> nx.Graph:
    """
    Memuat topologi jaringan dari file GML.

    Args:
        gml_file_path (str): Path lengkap ke file GML.

    Returns:
        nx.Graph: Objek NetworkX Graph yang merepresentasikan topologi.
                  Mengembalikan None jika file tidak ditemukan atau ada kesalahan.
    """
    if not os.path.exists(gml_file_path):
        print(f"Error: File GML tidak ditemukan di {gml_file_path}")
        return None
    try:
        # Memuat grafik dari file GML
        # networkx.read_gml secara otomatis menangani parsing GML
        graph = nx.read_gml(gml_file_path)

        # GML dapat memiliki node dengan ID string atau integer.
        # Untuk konsistensi, kita akan memastikan node ID adalah integer
        # dan berurutan dari 0 hingga N-1.
        # Kita juga akan mencoba menggunakan atribut 'label' sebagai nama node jika ada.

        # Buat pemetaan dari ID asli ke ID integer baru yang berurutan
        # Ini penting untuk konsistensi dengan DRL/GA yang mungkin mengasumsikan
        # node ID berbasis indeks.
        original_nodes = list(graph.nodes())
        new_node_mapping = {original_id: i for i, original_id in enumerate(original_nodes)}

        # Buat grafik baru dengan node ID integer yang berurutan
        new_graph = nx.Graph()

        for original_id, new_id in new_node_mapping.items():
            node_attrs = graph.nodes[original_id]
            # Salin semua atribut node asli ke node baru
            new_graph.add_node(new_id, **node_attrs)
            # Simpan label asli jika ada, untuk referensi
            if 'label' in node_attrs:
                new_graph.nodes[new_id]['original_label'] = node_attrs['label']
            else:
                new_graph.nodes[new_id]['original_label'] = str(original_id)


        # Tambahkan edge ke grafik baru menggunakan ID node yang baru
        for u_original, v_original, data in graph.edges(data=True):
            u_new = new_node_mapping[u_original]
            v_new = new_node_mapping[v_original]
            new_graph.add_edge(u_new, v_new, **data) # Salin atribut edge

        print(f"Berhasil memuat topologi dari {gml_file_path}. Jumlah node: {new_graph.number_of_nodes()}, Jumlah edge: {new_graph.number_of_edges()}")
        return new_graph

    except Exception as e:
        print(f"Error saat memuat file GML {gml_file_path}: {e}")
        return None

def get_topology_zoo_path():
    """
    Mengembalikan path yang diharapkan untuk direktori Topology Zoo.
    Asumsi: file .graphml dari Topology Zoo (misal GtsCe.graphml)
    ditempatkan langsung di direktori proyek sdn_ai_placement.
    """
    return os.path.dirname(os.path.abspath(__file__))

# Contoh penggunaan (tidak perlu dijalankan sebagai skrip utama)
if __name__ == '__main__':
    # Untuk menjalankan contoh ini, pastikan GtsCe.graphml ada di direktori yang sama
    # dengan script ini.
    
    topology_dir = get_topology_zoo_path()
    sample_gml_file = os.path.join(topology_dir, "GtsCe.graphml")

    if os.path.exists(sample_gml_file):
        print(f"Mencoba memuat topologi: {sample_gml_file}")
        graph = load_topology_from_gml(sample_gml_file)

        if graph:
            print(f"Berhasil memuat topologi GtsCe. Node: {graph.nodes()}, Edges: {graph.edges()}")
            print(f"Jumlah node: {graph.number_of_nodes()}")
            print(f"Jumlah edge: {graph.number_of_edges()}")
            # Cetak beberapa label node asli untuk verifikasi
            print("Beberapa node asli dan ID baru:")
            for i, node_id in enumerate(graph.nodes()):
                if i < 5: # Cetak 5 node pertama
                    print(f"  Node ID baru: {node_id}, Label asli: {graph.nodes[node_id].get('original_label', 'N/A')}")
            
            # Anda bisa memvisualisasikan grafik jika mau (membutuhkan matplotlib)
            # import matplotlib.pyplot as plt
            # nx.draw(graph, with_labels=True, node_color='lightblue', node_size=500)
            # plt.show()
        else:
            print("Gagal memuat topologi GtsCe.")
    else:
        print(f"File contoh GML tidak ditemukan di {sample_gml_file}.")
        print("Silakan unduh GtsCe.graphml dari GitHub dan tempatkan di direktori proyek Anda.")