# sdn_ai_placement/traditional_placement.py

import networkx as nx

def find_central_nodes(graph: nx.Graph, num_controllers: int):
    """
    Menemukan node paling sentral dalam grafik berdasarkan closeness centrality.
    Ini adalah metode tradisional untuk penempatan controller.

    Args:
        graph (nx.Graph): Objek NetworkX yang merepresentasikan topologi.
        num_controllers (int): Jumlah controller yang akan ditempatkan.

    Returns:
        list: Daftar node ID yang dipilih sebagai lokasi controller.
    """
    if not graph.nodes():
        return []

    if num_controllers <= 0:
        return []

    # Hitung closeness centrality untuk setiap node
    # closeness_centrality = nx.closeness_centrality(graph)
    # Untuk grafik yang tidak terhubung, closeness centrality tidak terdefinisi dengan baik.
    # Kita bisa menggunakan degree centrality sebagai alternatif sederhana
    # atau memastikan grafik terhubung. Untuk kesederhanaan, mari gunakan degree centrality
    # atau pastikan kita hanya memilih dari komponen terbesar yang terhubung.

    # Menggunakan degree centrality sebagai pendekatan sederhana untuk "paling terhubung"
    # Ini lebih robust untuk grafik yang mungkin tidak terhubung sepenuhnya.
    degree_centrality = nx.degree_centrality(graph)

    # Urutkan node berdasarkan closeness centrality (atau degree centrality) secara menurun
    # dan ambil 'num_controllers' teratas.
    sorted_nodes = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)

    # Pilih node teratas sebagai lokasi controller
    controller_locations = [node for node, centrality in sorted_nodes[:num_controllers]]

    return controller_locations

# Contoh penggunaan (tidak perlu dijalankan sebagai skrip utama)
if __name__ == '__main__':
    # Buat topologi dummy
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (1, 4),
        (2, 5), (3, 6), (4, 7),
        (5, 8), (6, 8), (7, 8)
    ]) # Contoh topologi dengan node 1 dan 8 sebagai hub

    print("Topologi nodes:", G.nodes())
    print("Topologi edges:", G.edges())

    # Coba tempatkan 1 controller
    controllers_1 = find_central_nodes(G, 1)
    print(f"\nPenempatan Controller Tradisional (1 controller): {controllers_1}")

    # Coba tempatkan 2 controller
    controllers_2 = find_central_nodes(G, 2)
    print(f"Penempatan Controller Tradisional (2 controller): {controllers_2}")

    # Coba tempatkan 3 controller
    controllers_3 = find_central_nodes(G, 3)
    print(f"Penempatan Controller Tradisional (3 controller): {controllers_3}")

    # Topologi yang lebih kompleks
    G_complex = nx.Graph()
    G_complex.add_edges_from([
        ('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E'),
        ('D', 'F'), ('E', 'F'), ('F', 'G'), ('G', 'H')
    ])
    print("\nTopologi kompleks nodes:", G_complex.nodes())
    print("Topologi kompleks edges:", G_complex.edges())

    controllers_complex_2 = find_central_nodes(G_complex, 2)
    print(f"Penempatan Controller Tradisional (2 controller, kompleks): {controllers_complex_2}")