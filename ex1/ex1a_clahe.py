from pathlib import Path
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "saida"
SAIDA.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--imagem", default=str(RAIZ.parent / "dados" / "frontal_kitti.png"))
caminho = Path(ap.parse_args().imagem)

# imdecode + fromfile: funciona no Windows com acentos no caminho.
img = cv2.imdecode(np.fromfile(str(caminho), np.uint8), cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(caminho)

# 1) Metadados
h, w = img.shape[:2]
print("Resolução (L x A):", w, "x", h)
print("dtype:", img.dtype)
print("Canais:", img.shape[2] if img.ndim == 3 else 1)
print("Tamanho em bytes (pixels na memória):", img.nbytes)
print("Tamanho em bytes (arquivo):", caminho.stat().st_size)

# 2) Brilho médio por quadrante (média do canal cinza, 0-255)
cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
my, mx = h // 2, w // 2
quadrantes = {"superior_esquerdo": cinza[:my, :mx], "superior_direito": cinza[:my, mx:],
              "inferior_esquerdo": cinza[my:, :mx], "inferior_direito": cinza[my:, mx:]}
for nome, q in quadrantes.items():
    print(f"Brilho médio {nome}: {q.mean():.2f}")

# 3) CLAHE: equalização adaptativa por blocos 8x8 com limite de contraste 2.0
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
eq = clahe.apply(cinza)

# COMENTÁRIO TÉCNICO - iluminação x percepção:
# Sombras (árvores, viadutos, túneis) reduzem o contraste faixa/asfalto e
# obstáculo/fundo; limiares fixos e detectores de borda passam a falhar.
# Ofuscamento (sol baixo, faróis) satura pixels em 255 e apaga detalhes,
# podendo esconder faixas e pedestres. A diferença de brilho entre quadrantes
# (ex.: um lado em sombra e o outro iluminado) mostra que uma correção global não
# basta: o CLAHE equaliza localmente, recupera contraste nas sombras, e o
# clipLimit evita amplificar demais o ruído (que ainda pode aumentar).

# 4) Histogramas antes/depois no mesmo gráfico
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.bincount(cinza.ravel(), minlength=256) / cinza.size, label="Antes")
ax.plot(np.bincount(eq.ravel(), minlength=256) / eq.size, label="Após CLAHE")
ax.set(title="Histograma antes x depois do CLAHE", xlabel="Intensidade (0-255)",
       ylabel="Frequência relativa", xlim=(0, 255))
ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(SAIDA / "histogramas.png", dpi=120)

# 5) Original e equalizada lado a lado
fig2, axs = plt.subplots(1, 2, figsize=(12, 4))
for a, im, t in zip(axs, [cinza, eq], ["Original (cinza)", "Após CLAHE"]):
    a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t); a.axis("off")
fig2.tight_layout()
fig2.savefig(SAIDA / "original_x_clahe.png", dpi=120)

print("Figuras salvas em:", SAIDA)
plt.show()
