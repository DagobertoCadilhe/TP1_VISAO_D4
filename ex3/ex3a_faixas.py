from pathlib import Path
import argparse
import os
import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "saida"
SAIDA.mkdir(exist_ok=True)
N_FRAMES = 50
LARGURA, ALTURA = 1280, 720

# 1) BEV: 4 pontos escolhidos manualmente (sup. esq., sup. dir., inf. dir., inf. esq.)
SRC = np.float32([[570, 430], [710, 430], [1150, 710], [130, 710]])   # trapézio da pista
DST = np.float32([[290, 0], [990, 0], [990, 719], [290, 719]])        # retângulo na BEV
M = cv2.getPerspectiveTransform(SRC, DST)
M_INV = cv2.getPerspectiveTransform(DST, SRC)                         # volta para a câmera


def mascara_hls_sobel(bev):
    """2) Segmentação na BEV: canal S do HLS combinado com Sobel X."""
    hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
    l, s = hls[:, :, 1], hls[:, :, 2]
    cinza = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3))   # bordas verticais (faixas)
    sobel = np.uint8(255 * gx / (gx.max() + 1e-9))
    amarelo = (s >= 90) & (l >= 75)          # S alto: faixa amarela
    branco = (l >= 165) & (s < 95)           # L alto: faixa branca
    mascara = np.uint8((amarelo | branco | (sobel >= 23)) * 255)
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def canny_hough(mascara):
    """3) Canny + HoughLinesP sobre a máscara."""
    bordas = cv2.Canny(mascara, 60, 150)
    linhas = cv2.HoughLinesP(bordas, 1, np.pi / 180, 35, minLineLength=35, maxLineGap=70)
    vista = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    if linhas is not None:
        for x1, y1, x2, y2 in linhas.reshape(-1, 4):
            cv2.line(vista, (x1, y1), (x2, y2), (0, 120, 255), 2)
    return linhas, vista


def sobrepor(frame, linhas):
    """4) Desenha as faixas na BEV e aplica a transformação inversa."""
    camada = np.zeros_like(frame)
    if linhas is not None:
        for x1, y1, x2, y2 in linhas.reshape(-1, 4):
            meio = (x1 + x2) / 2
            # mantém segmentos quase verticais e fora do centro da pista
            if abs(x2 - x1) <= 120 and (meio < 510 or meio > 770):
                cv2.line(camada, (x1, y1), (x2, y2), (0, 230, 0), 7)
    camada = cv2.warpPerspective(camada, M_INV, (LARGURA, ALTURA))
    return cv2.addWeighted(frame, 1, camada, 0.9, 0)


def painel(imagens, titulos):
    """Etapas lado a lado."""
    blocos = []
    for img, t in zip(imagens, titulos):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (480, 270))
        cv2.putText(img, t, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        blocos.append(img)
    return np.hstack(blocos)


def salvar(nome, img):
    # imencode + tofile: funciona no Windows com acentos no caminho.
    cv2.imencode(".png", img)[1].tofile(str(SAIDA / nome))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(RAIZ.parent / "dados" / "estrada_120_frames.mp4"))
    video = Path(ap.parse_args().video).resolve()
    # Abre pelo nome relativo para evitar falha do VideoCapture com acentos no Windows.
    os.chdir(video.parent)
    cap = cv2.VideoCapture(video.name)
    if not cap.isOpened():
        raise FileNotFoundError(video)

    for i in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Vídeo acabou no frame {i}; são necessários {N_FRAMES}.")
        frame = cv2.resize(frame, (LARGURA, ALTURA))
        bev = cv2.warpPerspective(frame, M, (LARGURA, ALTURA))
        mascara = mascara_hls_sobel(bev)
        linhas, vista_hough = canny_hough(mascara)
        final = sobrepor(frame, linhas)
        p = painel([bev, mascara, vista_hough], ["BEV", "HLS S + Sobel X", "Canny + Hough"])
        print(f"Frame {i + 1:02d}/{N_FRAMES} | segmentos Hough: {0 if linhas is None else len(linhas)}")
        if i in (0, N_FRAMES - 1):
            salvar(f"painel_frame_{i:02d}.png", p)
            salvar(f"final_frame_{i:02d}.png", final)
        cv2.imshow("Etapas", p)
        cv2.imshow("Faixas sobrepostas", final)
        if cv2.waitKey(30) & 0xFF == 27:   # ESC encerra
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"{i + 1} frames contínuos processados. Imagens em: {SAIDA}")
