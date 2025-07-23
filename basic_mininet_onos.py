# sdn_ai_placement/basic_mininet_onos.py

import os
import time
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info

def basic_onos_topology():
    """
    Membuat topologi Mininet sederhana (2 host, 1 switch)
    dan menghubungkannya ke controller ONOS eksternal.
    """
    setLogLevel('info')

    info('Memulai Mininet dengan controller ONOS...\n')

    # Inisialisasi Mininet dengan RemoteController
    # ONOS biasanya berjalan di localhost:6653
    net = Mininet(
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        switch=OVSKernelSwitch,
        link=TCLink, # Menggunakan TCLink untuk mendukung pengaturan bandwidth, delay, dll.
        cleanup=True # Membersihkan sisa-sisa proses Mininet sebelumnya
    )

    info('Menambahkan controller...\n')
    # Tambahkan controller secara eksplisit
    c0 = net.addController('c0')

    info('Menambahkan host dan switch...\n')
    # Tambahkan host
    h1 = net.addHost('h1', ip='10.0.0.1/24')
    h2 = net.addHost('h2', ip='10.0.0.2/24')

    # Tambahkan switch
    s1 = net.addSwitch('s1')

    info('Membuat link...\n')
    # Membuat link antara host dan switch
    # Anda bisa menambahkan parameter link seperti bandwidth (bw), delay, loss
    net.addLink(h1, s1, bw=10, delay='10ms', loss=0)
    net.addLink(h2, s1, bw=10, delay='10ms', loss=0)

    info('Memulai jaringan...\n')
    net.start()

    # Tunggu sebentar agar ONOS memiliki waktu untuk menemukan switch dan menginstal flow
    info('Menunggu ONOS untuk menginisialisasi dan menemukan switch...\n')
    time.sleep(10) # Berikan waktu 10 detik untuk ONOS

    info('Menguji konektivitas...\n')
    # Uji ping antara host
    result = h1.cmd('ping -c 3 %s' % h2.IP())
    info(result)

    # Anda bisa masuk ke CLI Mininet untuk interaksi manual
    info('Memulai Mininet CLI. Ketik "exit" untuk keluar.\n')
    CLI(net)

    info('Menghentikan jaringan Mininet...\n')
    net.stop()
    info('Jaringan Mininet dihentikan.\n')

if __name__ == '__main__':
    # Pastikan ONOS sedang berjalan sebelum menjalankan skrip ini
    # Anda bisa memulai ONOS dengan: onos-service start
    # Dan memverifikasi dengan: onos-service status
    print("Pastikan ONOS controller sudah berjalan (onos-service start) sebelum melanjutkan.")
    print("Tekan Enter untuk melanjutkan...")
    input() # Menunggu input pengguna

    # Jalankan topologi
    basic_onos_topology()