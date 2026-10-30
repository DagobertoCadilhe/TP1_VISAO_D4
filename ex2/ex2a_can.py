from pathlib import Path
import csv
import math
import multiprocessing as mp
import struct
import time
import can
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "saida"
CSV = SAIDA / "mensagens_can.csv"
HZ = 10          # 10 Hz -> período de 100 ms
CICLOS = 100     # 10 s de simulação

# ID -> (nome, unidade, escala). Valor físico = valor bruto * escala.
SINAIS = {0x100: ("velocidade", "km/h", 0.01),
          0x101: ("rpm", "rpm", 1.0),
          0x102: ("temperatura", "°C", 0.01)}
FILTROS = [{"can_id": i, "can_mask": 0x7FF, "extended": False} for i in SINAIS]


def valores(t):
    """Sinais simulados no instante t (s)."""
    return {0x100: 60 + 20 * math.sin(2 * math.pi * t / 5),
            0x101: 2000 + 800 * math.sin(2 * math.pi * t / 5),
            0x102: 80 + 0.5 * t}


def codificar(can_id, valor):
    # 2 bytes little-endian sem sinal (uint16).
    return struct.pack("<H", round(valor / SINAIS[can_id][2]))


def decodificar(msg):
    nome, unidade, escala = SINAIS[msg.arbitration_id]
    (bruto,) = struct.unpack("<H", msg.data)
    return nome, bruto * escala, unidade


def publicador(q, pronto):
    """Processo 1: publica 0x100, 0x101 e 0x102 a cada 100 ms."""
    pronto.wait(10)
    with can.Bus(interface="virtual", channel="pub") as tx, \
         can.Bus(interface="virtual", channel="pub") as gateway:
        t0 = time.monotonic()
        for k in range(CICLOS):
            time.sleep(max(0, t0 + k / HZ - time.monotonic()))   # agenda fixa de 10 Hz
            for can_id, v in valores(k / HZ).items():
                tx.send(can.Message(arbitration_id=can_id, data=codificar(can_id, v),
                                    is_extended_id=False))       # CAN 2.0A: ID de 11 bits
                m = gateway.recv(1)                              # sai do barramento
                q.put((m.arbitration_id, bytes(m.data), m.timestamp))  # ponte IPC
    q.put(None)  # fim


def assinante(q, pronto):
    """Processo 2: recebe, filtra por ID, decodifica, imprime e grava CSV."""
    with can.Bus(interface="virtual", channel="sub", preserve_timestamps=True) as gateway, \
         can.Bus(interface="virtual", channel="sub", can_filters=FILTROS) as rx, \
         open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "id", "sinal", "valor", "unidade"])
        pronto.set()
        while (item := q.get(timeout=20)) is not None:
            can_id, data, ts = item
            gateway.send(can.Message(arbitration_id=can_id, data=data, timestamp=ts,
                                     is_extended_id=False))
            msg = rx.recv(0.2)
            if msg is None:          # ID fora do filtro é descartado
                continue
            nome, valor, unidade = decodificar(msg)
            print(f"{msg.timestamp:.3f}  0x{msg.arbitration_id:03X}  {nome}={valor:.2f} {unidade}",
                  flush=True)
            w.writerow([f"{msg.timestamp:.6f}", f"0x{msg.arbitration_id:03X}", nome,
                        f"{valor:.2f}", unidade])


def plotar():
    with open(CSV, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    t0 = float(linhas[0]["timestamp"])
    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, (can_id, (nome, unidade, _)) in zip(axs, SINAIS.items()):
        r = [l for l in linhas if l["id"] == f"0x{can_id:03X}"]
        ax.plot([float(l["timestamp"]) - t0 for l in r], [float(l["valor"]) for l in r], ".-")
        ax.set_ylabel(f"{nome} ({unidade})"); ax.grid(alpha=.3)
    axs[0].set_title("Séries temporais CAN (10 Hz)")
    axs[-1].set_xlabel("Tempo (s)")
    fig.tight_layout()
    fig.savefig(SAIDA / "series_can.png", dpi=120)
    plt.show()


# COMENTÁRIO - CAN 2.0A x CAN FD:
# CAN 2.0A (clássico): identificador de 11 bits (2048 IDs), payload máximo de
# 8 bytes e taxa de até 1 Mbit/s em todo o quadro.
# CAN FD: mantém IDs de 11 (ou 29) bits, aceita payload de até 64 bytes e, com
# BRS, a fase de dados usa taxa maior (tipicamente 2 a 5 Mbit/s, até ~8),
# enquanto a arbitragem continua em até 1 Mbit/s. Resultado: mais dados úteis
# por quadro e menos overhead, importante para câmeras/radares/ADAS.
# Obs.: o VirtualBus não simula bitrate nem arbitragem.

if __name__ == "__main__":
    SAIDA.mkdir(exist_ok=True)
    ctx = mp.get_context("spawn")
    q, pronto = ctx.Queue(), ctx.Event()
    sub = ctx.Process(target=assinante, args=(q, pronto))
    pub = ctx.Process(target=publicador, args=(q, pronto))
    sub.start(); pub.start()
    pub.join(); sub.join()
    print("CSV salvo em:", CSV)
    plotar()
