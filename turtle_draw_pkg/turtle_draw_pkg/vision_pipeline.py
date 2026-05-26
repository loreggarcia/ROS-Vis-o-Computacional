"""Pipeline de visao do Turtle Draw.

Consome a imagem final de bordas ja exportada pelo notebook
(`edges_final.png`) e gera caminhos que se aproximam do grafico de
coordenadas logicas mostrado no notebook.
"""

from typing import List, Sequence, Tuple

import cv2
import numpy as np

Pixel = Tuple[int, int]
Ponto = Tuple[float, float]
Segmento = List[Ponto]


def gerar_caminho_a_partir_da_imagem(
    caminho_imagem: str,
    tamanho_minimo_componente: int = 2,
    manter_maiores_componentes: int = 500,
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
    # Transforma a imagem de bordas em segmentos desenhaveis.
    # Entrada: imagem binaria exportada pelo notebook.
    # Saida: segmentos em coordenadas do mundo e a imagem binaria ja limpa.

    # Esses parametros eram usados em uma versao anterior da pipeline.
    del tamanho_kernel, sigma, limiar_baixo, limiar_alto
    del tamanho_mundo, margem, salto_maximo

    # 1) Le a imagem exportada pelo notebook e garante formato binario.
    bordas_binarias = _carregar_bordas(caminho_imagem)
    if margem_remocao_borda > 0:
        # 2) Remove artefatos grudados na borda da imagem.
        bordas_binarias = _remover_moldura_externa(
            bordas_binarias,
            margem=max(1, margem_remocao_borda),
        )

    # 3) Descarta componentes muito pequenos para reduzir ruido.
    bordas_binarias = _filtrar_componentes(
        bordas_binarias,
        tamanho_minimo_componente=tamanho_minimo_componente,
        manter_maiores_componentes=manter_maiores_componentes,
    )

    # 4) Fecha pequenas falhas para transformar bordas finas em contornos mais continuos.
    bordas_binarias = _conectar_bordas(bordas_binarias)

    # 5) Extrai contornos continuos da imagem para formar trajetorias desenhaveis.
    segmentos_pixels = _segmentos_por_contorno(
        bordas_binarias,
        distancia_minima_entre_pontos=distancia_minima_entre_pontos,
        pontos_minimos_por_segmento=pontos_minimos_por_segmento,
    )
    segmentos_pixels = _ordenar_segmentos_para_desenho(segmentos_pixels)

    altura, largura = bordas_binarias.shape
    # 6) Mapeia coordenadas de pixel para o sistema de coordenadas do Turtlesim.
    segmentos_mundo = _mapear_para_turtlesim(
        segmentos_pixels,
        altura_imagem=altura,
        largura_imagem=largura,
    )

    return segmentos_mundo, bordas_binarias



def _carregar_bordas(caminho: str) -> np.ndarray:
    """Carrega a imagem de bordas e retorna como uint8 binario (0 ou 1)."""

    # Le a imagem exatamente como foi salva no notebook.
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
    # Isso simplifica o restante do pipeline para trabalhar so com fundo e borda.
    _, binaria = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)
    return binaria.astype(np.uint8)


def _remover_moldura_externa(bordas: np.ndarray, margem: int) -> np.ndarray:
    """Remove artefatos de moldura colados nas bordas da imagem."""

    resultado = bordas.copy()
    # Limita a margem para nao apagar uma parte excessiva do desenho.
    margem = min(margem, max(1, min(resultado.shape) // 4))
    resultado[:margem, :] = 0
    resultado[-margem:, :] = 0
    resultado[:, :margem] = 0
    resultado[:, -margem:] = 0
    return resultado


def _segmentos_por_varredura(
    bordas: np.ndarray,
    distancia_minima_entre_pontos: float,
    pontos_minimos_por_segmento: int,
) -> List[List[Pixel]]:
    """Varre a imagem linha a linha para refletir o grafico do notebook.

    Cada trecho horizontal contiguo de pixels ativos vira um segmento.
    Isso evita ligacoes artificiais entre regioes distantes e produz um
    caminho muito mais parecido com o scatter plot do notebook.
    """

    # Aqui a imagem deixa de ser tratada como matriz e passa a ser tratada como tracos.
    segmentos: List[List[Pixel]] = []

    # Percorre linha por linha para reproduzir a logica de varredura usada no notebook.
    for linha in range(bordas.shape[0]):
        colunas_ativas = np.where(bordas[linha] > 0)[0]
        if len(colunas_ativas) == 0:
            continue

        # Inicia um segmento com o primeiro pixel ativo da linha.
        atual: List[Pixel] = [(linha, int(colunas_ativas[0]))]
        for coluna in colunas_ativas[1:]:
            if int(coluna) == atual[-1][1] + 1:
                # Se o pixel atual encosta no anterior, ele faz parte do mesmo traco.
                atual.append((linha, int(coluna)))
                continue

            # Quando ha uma quebra, fecha o segmento atual e inicia outro.
            segmentos.extend(
                _finalizar_segmento_varrido(
                    atual,
                    distancia_minima_entre_pontos=distancia_minima_entre_pontos,
                    pontos_minimos_por_segmento=pontos_minimos_por_segmento,
                )
            )
            atual = [(linha, int(coluna))]

        segmentos.extend(
            _finalizar_segmento_varrido(
                atual,
                distancia_minima_entre_pontos=distancia_minima_entre_pontos,
                pontos_minimos_por_segmento=pontos_minimos_por_segmento,
            )
        )

    return segmentos


def _segmentos_por_contorno(
    bordas: np.ndarray,
    distancia_minima_entre_pontos: float,
    pontos_minimos_por_segmento: int,
) -> List[List[Pixel]]:
    """Extrai contornos reais da imagem para evitar tracos soltos por varredura."""

    # O OpenCV espera uma imagem 0/255 para detectar contornos.
    mascara = (bordas * 255).astype(np.uint8)
    contornos, _ = cv2.findContours(
        mascara,
        # Mantem todos os contornos para percorrer o desenho completo.
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    segmentos: List[List[Pixel]] = []
    for contorno in contornos:
        if len(contorno) == 0:
            continue

        # findContours retorna pontos como (x, y); aqui convertemos para (linha, coluna).
        segmento = [
            (int(ponto[0][1]), int(ponto[0][0]))
            for ponto in contorno
        ]
        segmento = _normalizar_inicio_segmento(segmento)
        segmentos.extend(
            _finalizar_segmento_varrido(
                segmento,
                distancia_minima_entre_pontos=distancia_minima_entre_pontos,
                pontos_minimos_por_segmento=pontos_minimos_por_segmento,
            )
        )

    return segmentos


def _conectar_bordas(bordas: np.ndarray) -> np.ndarray:
    """Fecha pequenas falhas e reforca tracos muito finos."""

    mascara = (bordas * 255).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    fechada = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)
    reforcada = cv2.dilate(fechada, kernel, iterations=1)
    return (reforcada > 0).astype(np.uint8)


def _normalizar_inicio_segmento(segmento: Sequence[Pixel]) -> List[Pixel]:
    """Faz cada segmento comecar no ponto mais alto e mais a esquerda."""

    if not segmento:
        return []

    melhor_indice = min(
        range(len(segmento)),
        key=lambda indice: (segmento[indice][0], segmento[indice][1]),
    )
    segmento_ordenado = list(segmento[melhor_indice:]) + list(segmento[:melhor_indice])

    # Escolhe a direcao cujo segundo ponto fica mais acima e mais a esquerda.
    if len(segmento_ordenado) >= 3:
        proximo = segmento_ordenado[1]
        anterior = segmento_ordenado[-1]
        if (anterior[0], anterior[1]) < (proximo[0], proximo[1]):
            ponto_inicial = segmento_ordenado[0]
            miolo_invertido = list(reversed(segmento_ordenado[1:]))
            segmento_ordenado = [ponto_inicial] + miolo_invertido

    return segmento_ordenado


def _ordenar_segmentos_para_desenho(segmentos: List[List[Pixel]]) -> List[List[Pixel]]:
    """Ordena segmentos de cima para baixo e da esquerda para a direita."""

    return sorted(
        segmentos,
        key=lambda segmento: (
            min(linha for linha, _ in segmento),
            min(coluna for _, coluna in segmento),
            len(segmento),
        ),
    )


def _filtrar_componentes(
    bordas: np.ndarray,
    tamanho_minimo_componente: int,
    manter_maiores_componentes: int | None,
) -> np.ndarray:
    """Mantem apenas componentes conectados relevantes da imagem binaria."""

    # Agrupa pixels conectados para separar tracos reais de ruido isolado.
    componentes = _rotular_componentes(bordas)
    componentes_validos = [
        componente
        for componente in componentes
        if len(componente) >= max(1, tamanho_minimo_componente)
    ]
    componentes_validos.sort(key=len, reverse=True)

    if manter_maiores_componentes is not None and manter_maiores_componentes > 0:
        # Mantem so os maiores grupos de pixels, descartando detalhes pequenos.
        componentes_validos = componentes_validos[:manter_maiores_componentes]

    filtrada = np.zeros_like(bordas, dtype=np.uint8)
    for componente in componentes_validos:
        # Reconstrui a imagem somente com os componentes aprovados.
        for linha, coluna in componente:
            filtrada[linha, coluna] = 1

    return filtrada


def _rotular_componentes(bordas: np.ndarray) -> List[List[Pixel]]:
    """Separa componentes 8-conectados para remover ruído isolado."""

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

            # BFS: parte de um pixel ativo e visita todos os vizinhos conectados.
            while indice < len(fila):
                atual_linha, atual_coluna = fila[indice]
                indice += 1
                componente.append((atual_linha, atual_coluna))

                # Busca os 8 vizinhos do pixel atual para montar um componente conectado.
                for delta_linha in (-1, 0, 1):
                    for delta_coluna in (-1, 0, 1):
                        if delta_linha == 0 and delta_coluna == 0:
                            continue

                        vizinho_linha = atual_linha + delta_linha
                        vizinho_coluna = atual_coluna + delta_coluna

                        if not (0 <= vizinho_linha < altura and 0 <= vizinho_coluna < largura):
                            continue

                        if bordas[vizinho_linha, vizinho_coluna] == 0:
                            continue

                        if visitados[vizinho_linha, vizinho_coluna]:
                            continue

                        visitados[vizinho_linha, vizinho_coluna] = True
                        fila.append((vizinho_linha, vizinho_coluna))

            componentes.append(componente)

    return componentes


def _finalizar_segmento_varrido(
    segmento: Sequence[Pixel],
    distancia_minima_entre_pontos: float,
    pontos_minimos_por_segmento: int,
) -> List[List[Pixel]]:
    """Aplica reducao minima a um segmento extraido por varredura."""

    # Remove pontos muito proximos para simplificar o caminho final.
    segmento_reduzido = _reduzir_rastro(
        segmento,
        max(distancia_minima_entre_pontos, 1.0) ** 2,
    )

    if len(segmento_reduzido) < max(2, pontos_minimos_por_segmento):
        # Segmentos curtos demais nao compensam um movimento de desenho.
        return []

    return [segmento_reduzido]


def _reduzir_rastro(segmento: Sequence[Pixel], limite_reducao2: float) -> List[Pixel]:
    """Remove redundancia sem perder a ordem pixel a pixel do traco."""

    if not segmento:
        return []

    reduzido: List[Pixel] = [segmento[0]]
    ul, uc = segmento[0]

    for linha, coluna in segmento[1:]:
        dl = linha - ul
        dc = coluna - uc
        # Ignora pontos perto demais do ultimo ponto mantido.
        if (dl * dl + dc * dc) < limite_reducao2:
            continue
        reduzido.append((linha, coluna))
        ul, uc = linha, coluna

    # Garante que o ultimo ponto do traco original nao seja perdido.
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

    # Descobre o retangulo minimo que envolve o desenho.
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
    # Usa a menor escala para preservar a proporcao e caber na arena.
    escala = min(escala_x, escala_y)

    # Centraliza o desenho dentro da arena do Turtlesim.
    arena = 11.0
    largura_desenhada = largura_util * escala
    altura_desenhada = altura_util * escala
    offset_x = (arena - largura_desenhada) / 2.0
    offset_y = (arena - altura_desenhada) / 2.0

    segmentos_mundo: List[Segmento] = []
    for segmento in segmentos_pixels:
        seg: Segmento = []
        for linha, coluna in segmento:
            # Converte coluna em x e inverte o eixo y da imagem para o mundo do simulador.
            tx = ((coluna - coluna_min) * escala) + offset_x
            ty = ((linha_max - linha) * escala) + offset_y
            seg.append((tx, ty))
        if seg:
            # Cada item da lista final representa um traco pronto para o draw_node executar.
            segmentos_mundo.append(seg)

    return segmentos_mundo
