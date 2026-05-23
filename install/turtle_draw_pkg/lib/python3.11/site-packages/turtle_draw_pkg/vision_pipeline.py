"""Pipeline de visao do Turtle Draw.

Consome a imagem final de bordas ja exportada pelo notebook
(`edges_final.png`) e gera caminhos seguindo os pixels ativos da imagem.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

Pixel = Tuple[int, int]
Ponto = Tuple[float, float]
Segmento = List[Ponto]

# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def gerar_caminho_a_partir_da_imagem(
    caminho_imagem: str,
    tamanho_minimo_componente: int = 1,
    manter_maiores_componentes: int = 10000,
    distancia_minima_entre_pontos: float = 1.0,
    tamanho_kernel: int = 3,
    sigma: float = 1.4,
    limiar_baixo: float = 0.4,
    limiar_alto: float = 0.3,
    margem_remocao_borda: int = 2,
    salto_maximo: int = 1,
    pontos_minimos_por_segmento: int = 1,
    tamanho_mundo: float = 11.0,
    margem: float = 1.0,
) -> Tuple[List[Segmento], np.ndarray]:
    """Carrega bordas pre-calculadas e retorna segmentos para o Turtlesim."""

    del tamanho_kernel, sigma, limiar_baixo, limiar_alto
    del manter_maiores_componentes, tamanho_mundo, margem

    bordas_binarias = _carregar_bordas(caminho_imagem)
    bordas_binarias = _remover_moldura_externa(
        bordas_binarias,
        margem=max(1, margem_remocao_borda),
    )
    segmentos_pixels = _rastrear_pixels_ativos(
        bordas_binarias,
        tamanho_minimo_componente=tamanho_minimo_componente,
        distancia_minima_entre_pontos=distancia_minima_entre_pontos,
        salto_maximo=salto_maximo,
        pontos_minimos_por_segmento=pontos_minimos_por_segmento,
    )

    altura, largura = bordas_binarias.shape
    segmentos_mundo = _mapear_para_turtlesim(
        segmentos_pixels,
        altura_imagem=altura,
        largura_imagem=largura,
    )

    return segmentos_mundo, bordas_binarias


# ---------------------------------------------------------------------------
# Funcoes internas
# ---------------------------------------------------------------------------

def _carregar_bordas(caminho: str) -> np.ndarray:
    """Carrega a imagem de bordas e retorna como uint8 binario (0 ou 1)."""

    img = cv2.imread(caminho, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(
            f'Nao foi possivel carregar a imagem em "{caminho}".\n'
            'Exporte do notebook com: cv2.imwrite("edges_final.png", bordas_binarias * 255)'
        )

    # Se for colorida ou com alpha, converte para cinza
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binariza: qualquer pixel nao-zero vira 1
    _, binaria = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)
    return binaria.astype(np.uint8)


def _remover_moldura_externa(bordas: np.ndarray, margem: int) -> np.ndarray:
    """Remove artefatos de moldura colados nas bordas da imagem."""

    resultado = bordas.copy()
    margem = min(margem, max(1, min(resultado.shape) // 4))
    resultado[:margem, :] = 0
    resultado[-margem:, :] = 0
    resultado[:, :margem] = 0
    resultado[:, -margem:] = 0
    return resultado


def _rastrear_pixels_ativos(
    bordas: np.ndarray,
    tamanho_minimo_componente: int,
    distancia_minima_entre_pontos: float,
    salto_maximo: int,
    pontos_minimos_por_segmento: int,
) -> List[List[Pixel]]:
    """Rastreia os pixels ativos de cima para baixo com conectividade 8."""

    visitados = np.zeros_like(bordas, dtype=bool)
    ativos = np.argwhere(bordas > 0)
    if len(ativos) == 0:
        return []

    segmentos: List[List[Pixel]] = []
    limite_reducao2 = max(distancia_minima_entre_pontos, 1.0) ** 2
    limite_salto2 = max(salto_maximo, 1) ** 2
    componentes = _rotular_componentes(bordas)

    componentes = [c for c in componentes if len(c) >= max(1, tamanho_minimo_componente)]
    componentes.sort(key=_chave_componente)

    for componente in componentes:
        while True:
            inicio = _proximo_pixel_superior_do_componente(componente, visitados)
            if inicio is None:
                break

            segmento = _seguir_traco(inicio, bordas, visitados, limite_salto2)
            if len(segmento) < max(tamanho_minimo_componente, pontos_minimos_por_segmento):
                continue

            segmento_reduzido = _reduzir_rastro(segmento, limite_reducao2)
            if len(segmento_reduzido) >= max(2, pontos_minimos_por_segmento):
                segmentos.append(segmento_reduzido)

    return segmentos


def _rotular_componentes(bordas: np.ndarray) -> List[List[Pixel]]:
    """Separa pixels ativos em componentes 8-conectados."""

    altura, largura = bordas.shape
    visitados = np.zeros_like(bordas, dtype=bool)
    componentes: List[List[Pixel]] = []

    for linha in range(altura):
        for coluna in range(largura):
            if bordas[linha, coluna] == 0 or visitados[linha, coluna]:
                continue

            fila: List[Pixel] = [(linha, coluna)]
            visitados[linha, coluna] = True
            componente: List[Pixel] = []
            indice = 0

            while indice < len(fila):
                atual = fila[indice]
                indice += 1
                componente.append(atual)

                for delta_linha in (-1, 0, 1):
                    for delta_coluna in (-1, 0, 1):
                        if delta_linha == 0 and delta_coluna == 0:
                            continue

                        nova_linha = atual[0] + delta_linha
                        nova_coluna = atual[1] + delta_coluna

                        if (
                            nova_linha < 0
                            or nova_linha >= altura
                            or nova_coluna < 0
                            or nova_coluna >= largura
                        ):
                            continue

                        if bordas[nova_linha, nova_coluna] == 0 or visitados[nova_linha, nova_coluna]:
                            continue

                        visitados[nova_linha, nova_coluna] = True
                        fila.append((nova_linha, nova_coluna))

            componentes.append(componente)

    return componentes


def _chave_componente(componente: Sequence[Pixel]) -> tuple[int, int, int]:
    """Ordena componentes de cima para baixo, priorizando os maiores."""

    linhas = [linha for linha, _ in componente]
    colunas = [coluna for _, coluna in componente]
    return (min(linhas), min(colunas), -len(componente))


def _proximo_pixel_superior_do_componente(
    componente: Sequence[Pixel],
    visitados: np.ndarray,
) -> Pixel | None:
    """Retorna o pixel nao visitado mais alto e a esquerda do componente."""

    candidatos = [(linha, coluna) for linha, coluna in componente if not visitados[linha, coluna]]
    if not candidatos:
        return None
    candidatos.sort()
    return candidatos[0]


def _seguir_traco(
    inicio: Pixel,
    bordas: np.ndarray,
    visitados: np.ndarray,
    limite_salto2: int,
) -> List[Pixel]:
    """Segue um traco pixel a pixel escolhendo sempre o melhor vizinho local."""

    segmento: List[Pixel] = [inicio]
    visitados[inicio[0], inicio[1]] = True
    atual = inicio
    direcao_anterior = (0, 0)

    while True:
        proximo = _escolher_proximo_vizinho(
            atual,
            bordas,
            visitados,
            direcao_anterior,
            limite_salto2,
        )
        if proximo is None:
            break

        segmento.append(proximo)
        visitados[proximo[0], proximo[1]] = True
        direcao_anterior = (proximo[0] - atual[0], proximo[1] - atual[1])
        atual = proximo

    return segmento


def _escolher_proximo_vizinho(
    atual: Pixel,
    bordas: np.ndarray,
    visitados: np.ndarray,
    direcao_anterior: Pixel,
    limite_salto2: int,
) -> Pixel | None:
    """Escolhe o proximo pixel por conectividade 8 e continuidade direcional."""

    candidatos: List[tuple[tuple[int, int, int, int], Pixel]] = []

    for delta_linha in (-1, 0, 1):
        for delta_coluna in (-1, 0, 1):
            if delta_linha == 0 and delta_coluna == 0:
                continue

            linha = atual[0] + delta_linha
            coluna = atual[1] + delta_coluna

            if (
                linha < 0
                or linha >= bordas.shape[0]
                or coluna < 0
                or coluna >= bordas.shape[1]
            ):
                continue

            if bordas[linha, coluna] == 0 or visitados[linha, coluna]:
                continue

            distancia2 = (delta_linha * delta_linha) + (delta_coluna * delta_coluna)
            if distancia2 > limite_salto2:
                continue

            # Prioriza continuidade do movimento; em empate, segue ordem raster.
            penalidade_direcao = abs(delta_linha - direcao_anterior[0]) + abs(delta_coluna - direcao_anterior[1])
            chave = (penalidade_direcao, distancia2, linha, coluna)
            candidatos.append((chave, (linha, coluna)))

    if not candidatos:
        return None

    candidatos.sort(key=lambda item: item[0])
    return candidatos[0][1]


def _reduzir_rastro(segmento: Sequence[Pixel], limite_reducao2: float) -> List[Pixel]:
    """Remove redundancia sem perder a ordem pixel a pixel do traco."""

    if not segmento:
        return []

    reduzido: List[Pixel] = [segmento[0]]
    ul, uc = segmento[0]

    for linha, coluna in segmento[1:]:
        dl = linha - ul
        dc = coluna - uc
        if (dl * dl + dc * dc) < limite_reducao2:
            continue
        reduzido.append((linha, coluna))
        ul, uc = linha, coluna

    if reduzido[-1] != segmento[-1]:
        reduzido.append(segmento[-1])

    return reduzido


def _mapear_para_turtlesim(
    segmentos_pixels: List[List[Pixel]],
    altura_imagem: int,
    largura_imagem: int,
    tamanho_turtlesim: float = 10.0,
) -> List[Segmento]:
    """Mapeia pixels para coordenadas do Turtlesim ampliando a area util."""

    if not segmentos_pixels:
        return []

    linhas = [linha for segmento in segmentos_pixels for linha, _ in segmento]
    colunas = [coluna for segmento in segmentos_pixels for _, coluna in segmento]
    linha_min = min(linhas)
    linha_max = max(linhas)
    coluna_min = min(colunas)
    coluna_max = max(colunas)

    largura_util = max((coluna_max - coluna_min) + 1, 1)
    altura_util = max((linha_max - linha_min) + 1, 1)

    escala_x = tamanho_turtlesim / largura_util
    escala_y = tamanho_turtlesim / altura_util
    escala = min(escala_x, escala_y)

    arena = 11.0
    largura_desenhada = largura_util * escala
    altura_desenhada = altura_util * escala
    offset_x = (arena - largura_desenhada) / 2.0
    offset_y = (arena - altura_desenhada) / 2.0

    segmentos_mundo: List[Segmento] = []
    for segmento in segmentos_pixels:
        seg: Segmento = []
        for linha, coluna in segmento:
            tx = ((coluna - coluna_min) * escala) + offset_x
            ty = ((linha_max - linha) * escala) + offset_y
            seg.append((tx, ty))
        if seg:
            segmentos_mundo.append(seg)

    return segmentos_mundo
