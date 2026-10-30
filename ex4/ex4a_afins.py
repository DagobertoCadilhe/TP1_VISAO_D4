from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "saida"
SAIDA.mkdir(exist_ok=True)
caminho = RAIZ.parent / "dados" / "frontal_kitti.png"
img = cv2.imdecode(np.fromfile(str(caminho), np.uint8), cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(caminho)
h, w = img.shape[:2]
centro = (w / 2, h / 2)
np.set_printoptions(suppress=True)


def warp(M):
    return cv2.warpAffine(img, np.float32(M), (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


def H(M):
    """2x3 -> 3x3 homogênea (última linha [0 0 1]) para compor por multiplicação."""
    return np.vstack([M, [0, 0, 1]])


# 1) Translação: x' = x + 80, y' = y + 30
# Uso: augmentation que simula deslocamento lateral do veículo na faixa /
# câmera montada fora do centro (bounding boxes devem ser deslocadas junto).
T = np.float32([[1, 0, 80], [0, 1, 30]])

# 2) Rotação de 10° em torno do centro
# Uso: robustez a roll da câmera (lombadas, inclinação da via, montagem
# torta). Usar ângulos pequenos (±10°): a estrada não aparece de cabeça para baixo.
R = cv2.getRotationMatrix2D(centro, 10, 1.0)

# 3) Escala (zoom 1.2x centrado)
# Uso: simular objetos mais próximos/distantes e variar o tamanho aparente de
# veículos e placas; também é a base do resize para a entrada fixa da rede.
S = cv2.getRotationMatrix2D(centro, 0, 1.2)

# 4) Afim customizada 2x3 (cisalhamento + escala + translação)
# Uso: aproximar pequenas mudanças de ponto de vista/altura da câmera e
# alinhar imagens de câmeras diferentes (afim preserva paralelismo, não perspectiva).
A = np.float32([[1.0, 0.15, -30], [0.02, 0.95, 10]])

transf = {"Translação (+80,+30)": T, "Rotação 10° centro": R,
          "Escala 1.2x": S, "Afim customizada": A}
for nome, Mx in transf.items():
    print(f"\n{nome} - matriz 2x3:\n{np.round(Mx, 4)}")

# 5) Composição: primeiro translação, depois rotação -> C = R · T (3x3)
C = H(R) @ H(T)
print("\nComposição R @ T (3x3):\n", np.round(C, 4))
composta = warp(C[:2])                 # um único warp (uma interpolação)
sequencial = cv2.warpAffine(warp(T), R, (w, h))   # dois warps em sequência
# A diferença vem do recorte/interpolação intermediários do caminho sequencial.
print("Diferença média composta x sequencial (0-255):",
      round(float(np.abs(composta.astype(int) - sequencial.astype(int)).mean()), 3))
# A ordem importa: T @ R gera outra matriz (matrizes não comutam).
print("R@T == T@R ?", np.allclose(C, H(T) @ H(R)))

# Exibição
imgs = [("Original", img)] + [(n, warp(Mx)) for n, Mx in transf.items()] + \
       [("Composta R·T (1 warp)", composta)]
fig, axs = plt.subplots(3, 2, figsize=(12, 7))
for ax, (t, im) in zip(axs.ravel(), imgs):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(t); ax.axis("off")
fig.tight_layout()
fig.savefig(SAIDA / "transformacoes.png", dpi=110)
print("\nFigura salva em:", SAIDA / "transformacoes.png")
plt.show()
