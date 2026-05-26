# Relatório Técnico Completo do Código

## Objetivo

&emsp;O projeto foi organizado em três arquivos principais:

1. `processamento.ipynb`
2. `turtle_draw_pkg/turtle_draw_pkg/vision_pipeline.py`
3. `turtle_draw_pkg/turtle_draw_pkg/draw_node.py`

&emsp;Os três formam uma cadeia:

- o notebook processa a imagem e gera a versão final de bordas;
- a pipeline converte essa imagem em segmentos no espaço do Turtlesim;
- o nó ROS 2 executa esses segmentos controlando a tartaruga.

&emsp;Este documento explica o código inteiro desses três arquivos, não apenas as decisões de arquitetura.

---

## 1. `processamento.ipynb`

### Papel do arquivo

&emsp;O notebook é a etapa de experimentação e validação visual do processamento. Ele não controla a tartaruga. O papel dele é preparar e conferir a imagem que depois será usada no ROS.

&emsp;O artefato mais importante produzido por esse arquivo é `edges_final.png`, exportado a partir da matriz final de bordas.

### Estrutura geral do notebook

&emsp;O notebook segue uma ordem clássica de processamento de imagem:

1. leitura da imagem original;
2. conversão para cinza;
3. suavização gaussiana;
4. cálculo de gradientes com Sobel;
5. magnitude e direção do gradiente;
6. supressão não-máxima;
7. limiarização dupla;
8. obtenção da imagem binária final;
9. visualização no espaço lógico do Turtlesim.

### Imports

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```


- `cv2`: leitura e escrita de imagem;
- `numpy`: operações matriciais;
- `matplotlib`: exibição dos resultados intermediários e finais.

### Leitura da imagem e conversão para cinza

&emsp; Basicamente, imagens coloridas possuem três canais, R (red), G (green) e B (blue). Em muitas aplicações de visão computacional, como detecção de bordas, não é necessário manter a informação de cor. Por isso, a imagem pode ser convertida para escala de cinza, passando a ter apenas um canal de intensidade.
Essa conversão traz duas vantagens principais:
&emsp;reduz o custo computacional, porque o processamento deixa de operar sobre três canais e passa a operar sobre apenas um;
&emsp;simplifica os algoritmos de detecção de bordas, já que eles dependem principalmente de variações de intensidade, e não de cor.
Existem diferentes formas de fazer essa conversão:
1. Média simples

2. Luminância (BT.601)

3. Luminância (BT.709)

&emsp;No código deste projeto foi usada a luminância BT.601, porque ela é uma formulação clássica, amplamente utilizada em processamento de imagem, e está alinhada ao comportamento esperado em bibliotecas como o OpenCV ao trabalhar com conversões tradicionais de intensidade.


```python
img_bgr = cv2.imread('dog.jpg')
img_colorida = img_bgr[:, :, ::-1]
R = img_colorida[:, :, 0]
G = img_colorida[:, :, 1]
B = img_colorida[:, :, 2]
img_cinza = 0.299 * R + 0.587 * G + 0.114 * B
```

- `cv2.imread(...)` lê a imagem no formato BGR;
- `[:, :, ::-1]` inverte os canais para RGB;
- os três canais são separados em `R`, `G` e `B`;
- a fórmula BT.601 gera uma imagem de intensidade única.

### Criação do kernel gaussiano
&emsp;O kernel gaussiano é uma pequena matriz usada para suavizar a imagem antes de aplicar outras operações, como detecção de bordas. A ideia é fazer com que cada pixel passe a ser calculado como uma média ponderada dos pixels vizinhos. Os pixels mais próximos do centro recebem peso maior, e os mais distantes recebem peso menor. Essa distribuição de pesos segue a função gaussiana, que tem formato de sino.

Na prática, isso serve para:

- reduzir ruídos da imagem;
- eliminar variações muito bruscas causadas por detalhes irrelevantes;
- preparar melhor a imagem para operadores como Sobel ou Canny.


```python
def criar_kernel(tm, sigma):
    k = tm // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    t1 = 1.0 / (2.0 * np.pi * sigma**2)
    t2 = np.exp(-((x**2) + (y**2)) / (2.0 * sigma**2))
    kernel = t1 * t2
    return kernel / kernel.sum()
```


- `tm` é o tamanho do kernel;
- `sigma` controla o espalhamento da gaussiana;
- `x` e `y` formam a grade 2D;
- `t1` e `t2` constroem a gaussiana;
- `kernel / kernel.sum()` normaliza a soma para 1.


### Convolução manual
&emsp;A convolução manual é o processo de aplicar um filtro sobre a imagem sem usar uma função pronta da biblioteca. Em vez disso, o próprio código percorre a imagem pixel por pixel e calcula o novo valor de cada posição.

A lógica é esta:

- escolher um kernel;
- centralizar esse kernel sobre um pixel da imagem;
- multiplicar cada valor do kernel pelos pixels correspondentes da vizinhança;
- somar os resultados;
- usar essa soma como novo valor do pixel filtrado.

```python
def convulacao(img, kernel):
    img_pad = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    for i in range(h_img):
        for j in range(l_img):
            regiao = img_pad[i:i + h_k, j:j + l_k]
            resultado[i, j] = np.sum(regiao * kernel)
```

### Filtros de Sobel
&emsp;Filtros de Sobel são operadores usados para detectar bordas por meio do cálculo do gradiente da imagem. Eles medem o quanto a intensidade varia em cada direção.

```python
sobel_x = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    dtype=np.float32,
)
sobel_y = np.array(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]],
    dtype=np.float32,
)
```

Esses kernels calculam derivadas aproximadas:

- `sobel_x`: variação horizontal;
- `sobel_y`: variação vertical.

&emsp; Os valores das matrizes são pesos fixos do filtro Sobel. Esses pesos foram projetados para medir variações de intensidade entre pixels vizinhos.


### Magnitude e ângulo do gradiente

&emsp;A magnitude e o ângulo do gradiente são calculados depois da aplicação dos filtros de Sobel, e servem para descrever a força e a direção das bordas na imagem.
Com os valores obtidos pelos filtros de Sobel, calcula-se a **magnitude do gradiente**:

&emsp;$$M = \sqrt{G_x^2 + G_y^2}$$

&emsp;Ela indica **quão forte é a borda** naquele ponto. Quanto maior a magnitude, maior é a variação de intensidade entre pixels vizinhos e, portanto, maior a probabilidade de existir uma borda real na imagem. Em seguida, calcula-se o **ângulo do gradiente**:

&emsp;$$ \theta = \arctan2(G_y, G_x)$$

&emsp;Esse ângulo indica a **direção da maior variação de intensidade**. Em outras palavras, ele mostra para que lado a intensidade da imagem está mudando com mais força naquele ponto.

```python
gradiente_x = filtro(img_suavizada, sobel_x)
gradiente_y = filtro(img_suavizada, sobel_y)
magnitude = np.sqrt((gradiente_x**2) + (gradiente_y**2))
angulo = np.arctan2(gradiente_y, gradiente_x)
angulo = np.rad2deg(angulo)
angulo[angulo < 0] += 180
```

- `gradiente_x` e `gradiente_y` medem direção local de maior variação;
- `magnitude` mede a intensidade da borda;
- `angulo` guarda a orientação da borda em graus no intervalo `[0, 180]`

### Supressão não-máxima


&emsp;Essas duas informações são importantes porque:

- a **magnitude** mostra onde a borda é mais intensa;
- o **ângulo** mostra a orientação dessa borda;
- juntas, elas são usadas nas próximas etapas do algoritmo, como a **supressão não-máxima**.

```python
def supressao(magnitude, angulo):
    if (0 <= angulo[i, j] < 22.5) or (157.5 <= angulo[i, j] <= 180):
        q = magnitude[i, j + 1]
        r = magnitude[i, j - 1]
```

&emsp;Essa função percorre a matriz de magnitudes e compara o pixel corrente com os vizinhos alinhados à direção do gradiente. Se o pixel atual não for o maior naquela direção, ele é zerado. Por fim, as bordas ficam mais finas e mais parecidas com contornos reais.

```python
if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
    resultado[i, j] = magnitude[i, j]
else:
    resultado[i, j] = 0
```


### Limiarização dupla

&emsp;A **limiarização dupla** é a etapa usada para separar os pixels que representam bordas fortes, bordas fracas e fundo. Em vez de usar apenas um único limiar, esse método utiliza **dois valores de referência**:

- um **limiar alto**;
- um **limiar baixo**.

&emsp;No código, esses limiares são calculados a partir do maior valor presente na imagem:

&emsp;$$\text{limiar\_alto} = \max(\text{img}) \cdot \text{limiar\_alto\_ratio}$$

&emsp;$$\text{limiar\_baixo} = \text{limiar\_alto} \cdot \text{limiar\_baixo\_ratio}$$

&emsp;Depois disso, cada pixel é classificado em uma das três categorias:

- **borda forte**: quando o valor do pixel é maior ou igual ao limiar alto;
- **borda fraca**: quando o valor está entre o limiar baixo e o limiar alto;
- **fundo**: quando o valor está abaixo do limiar baixo.

No código, isso aparece na lógica:

```python
forte_i, forte_j = np.where(img >= limiar_alto)
fraco_i, fraco_j = np.where((img < limiar_alto) & (img >= limiar_baixo))


def limiarizacao(img, limiar_baixo_ratio=0.4, limiar_alto_ratio=0.3):
    limiar_alto = img.max() * limiar_alto_ratio
    limiar_baixo = limiar_alto * limiar_baixo_ratio
```

Depois o código classifica pixels como fortes ou fracos:

```python
forte_i, forte_j = np.where(img >= limiar_alto)
fraco_i, fraco_j = np.where((img < limiar_alto) & (img >= limiar_baixo))
```

### Imagem binária final

Esse passo converte a imagem em:

- `1` para borda forte;
- `0` para fundo.

```python
bordas_binarias = (img_limiarizada == 255).astype(np.uint8)
```


### Remoção de moldura

&emsp;A necessidade dessa etapa surgiu porque, em algumas execuções, a imagem exportada continha artefatos nas extremidades, como pequenos traços ou contornos encostados na borda. Quando esses pixels eram enviados para o Turtlesim, eles geravam linhas grandes e incorretas, que não tinham relação com o cachorro.

```python
bordas_binarias[:margem_remocao_borda, :] = 0
bordas_binarias[-margem_remocao_borda:, :] = 0
bordas_binarias[:, :margem_remocao_borda] = 0
bordas_binarias[:, -margem_remocao_borda:] = 0
```

### Construção dos segmentos lógicos

&emsp;Depois que a imagem binária final é obtida e limpa, o próximo passo é transformá-la em **segmentos lógicos de desenho**. Esses segmentos representam conjuntos ordenados de pontos que depois poderão ser convertidos em coordenadas do Turtlesim.

&emsp;A ideia principal dessa etapa é deixar de tratar a imagem apenas como uma matriz de pixels e passar a tratá-la como um **conjunto de trajetos**. No código, isso começa com uma varredura linha por linha da imagem:

```python
for linha in range(bordas_binarias.shape[0]):
    colunas_ativas = np.where(bordas_binarias[linha] > 0)[0]
```

### Mapeamento para o espaço do Turtlesim

&emsp;Depois de construir os segmentos a partir da imagem binária, é necessário converter essas coordenadas de pixel para o sistema de coordenadas do Turtlesim. Para isso, o código identifica a área útil ocupada pelo desenho, calcula uma escala que preserve sua proporção e aplica deslocamentos para centralizá-lo na arena.

&emsp;Durante essa conversão, o eixo vertical também é invertido, porque na imagem os pixels crescem de cima para baixo, enquanto no Turtlesim o eixo Y cresce de baixo para cima. O resultado é um conjunto de segmentos posicionados corretamente no espaço lógico da simulação, prontos para serem percorridos pela tartaruga.

```python
tx = ((img_x - coluna_min) * escala) + offset_x
ty = ((linha_max - img_y) * escala) + offset_y
```

- `coluna_min` e `linha_max` servem como referência do bounding box útil;
- `escala` adapta o tamanho do desenho à arena;
- `offset_x` e `offset_y` centralizam a imagem;
- `linha_max - img_y` inverte o eixo vertical da imagem para o sistema cartesiano do Turtlesim.

### Plotagem final

&emsp;Na etapa final do notebook, os segmentos já convertidos para o espaço lógico são desenhados com `matplotlib`. Em vez de apenas exibir pixels isolados, o código usa linhas para representar os trajetos que serão percorridos pela tartaruga.

&emsp;Essa escolha é importante porque a visualização deixa de mostrar somente a imagem processada e passa a representar o caminho lógico do desenho. Assim, o gráfico final fica mais próximo do comportamento esperado no Turtlesim e facilita a comparação entre o resultado do processamento e o trajeto executado pelo nó ROS.

```python
for caminho_x, caminho_y in segmentos_turtlesim:
    plt.plot(caminho_x, caminho_y, c='blue', linewidth=1)
```



---

## 2. `vision_pipeline.py`

### Papel do arquivo

&emsp;Esse módulo não executa mais o detector de bordas inteiro em runtime. Ele consome diretamente a imagem final já pronta e a transforma em segmentos compatíveis com o espaço do Turtlesim. Ele retorna dois objetos:

- `segmentos_mundo`: lista de trajetos em coordenadas contínuas;
- `bordas_binarias`: imagem binária realmente utilizada.

### Cabeçalho e imports

```python
from typing import List, Sequence, Tuple

import cv2
import numpy as np
```

- `typing` melhora legibilidade e documentação interna;
- `cv2` faz leitura e binarização da imagem;
- `numpy` manipula a matriz de pixels.

### Aliases de tipo


```python
Pixel = Tuple[int, int]
Ponto = Tuple[float, float]
Segmento = List[Ponto]
```

- `Pixel`: coordenada discreta da imagem, no formato `(linha, coluna)`;
- `Ponto`: coordenada contínua no mundo do Turtlesim, no formato `(x, y)`;
- `Segmento`: uma sequência de pontos contínuos que será desenhada de uma vez.

### `gerar_caminho_a_partir_da_imagem(...)`

&emsp;Essa é a função pública principal do módulo.

```python
def gerar_caminho_a_partir_da_imagem(
    caminho_imagem: str,
    tamanho_minimo_componente: int = 20,
    manter_maiores_componentes: int = 80,
    distancia_minima_entre_pontos: float = 1.0,
    ...
) -> Tuple[List[Segmento], np.ndarray]:
```

&emsp;Parâmetros principais:

- `caminho_imagem`: caminho do `edges_final.png`;
- `tamanho_minimo_componente`: tamanho mínimo de uma componente conectada válida;
- `manter_maiores_componentes`: quantidade máxima de componentes preservadas;
- `distancia_minima_entre_pontos`: controle de redução de redundância.


```python
del tamanho_kernel, sigma, limiar_baixo, limiar_alto
del tamanho_mundo, margem, salto_maximo
```

&emsp;Esses parâmetros ficaram na assinatura por compatibilidade com versões anteriores do projeto, mas hoje não participam da lógica. O `del` explicita que foram mantidos apenas para não quebrar chamadas já existentes.

```python
bordas_binarias = _carregar_bordas(caminho_imagem)
```

&emsp;Primeiro, o arquivo binário é lido.

```python
if margem_remocao_borda > 0:
    bordas_binarias = _remover_moldura_externa(...)
```

&emsp;Depois, remove-se ruído de moldura.

```python
bordas_binarias = _filtrar_componentes(...)
```

&emsp;Em seguida, a imagem é limpa com base em conectividade.


```python
segmentos_pixels = _segmentos_por_varredura(...)
```

&emsp;Os pixels ativos viram segmentos discretos.

```python
segmentos_mundo = _mapear_para_turtlesim(...)
```

&emsp;Por fim, os segmentos são convertidos para coordenadas do mundo.

### `_carregar_bordas(...)`


```python
img = cv2.imread(caminho, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(...)
```

&emsp;Esse bloco valida se a imagem existe.

```python
if img.ndim == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

&emsp;Se a imagem estiver em RGB/BGR ou com alpha, ela é convertida para cinza.

```python
_, binaria = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)
return binaria.astype(np.uint8)
```

&emsp;Todo valor não nulo vira `1`.


- a pipeline usa uma imagem binária mínima, não uma escala de cinza ou gradiente.

### `_remover_moldura_externa(...)`


```python
resultado = bordas.copy()
margem = min(margem, max(1, min(resultado.shape) // 4))
resultado[:margem, :] = 0
resultado[-margem:, :] = 0
resultado[:, :margem] = 0
resultado[:, -margem:] = 0
```


### `_segmentos_por_varredura(...)`


```python
for linha in range(bordas.shape[0]):
    colunas_ativas = np.where(bordas[linha] > 0)[0]
```

&emsp;Essa função percorre a imagem linha por linha e busca pixels ativos.


```python
atual: List[Pixel] = [(linha, int(colunas_ativas[0]))]
```

&emsp;Se a linha contém pixels ativos, o primeiro pixel cria o segmento atual.


```python
if int(coluna) == atual[-1][1] + 1:
    atual.append((linha, int(coluna)))
    continue
```

&emsp;Se o próximo pixel está exatamente na coluna seguinte, ele pertence ao mesmo trecho horizontal.

```python
segmentos.extend(
    _finalizar_segmento_varrido(...)
)
atual = [(linha, int(coluna))]
```

### `_filtrar_componentes(...)`


```python
componentes = _rotular_componentes(bordas)
```

&emsp;Primeiro o código separa a imagem em componentes conectadas.

```python
componentes_validos = [
    componente
    for componente in componentes
    if len(componente) >= max(1, tamanho_minimo_componente)
]
```

&emsp;Aqui ruídos muito pequenos são descartados.


```python
componentes_validos.sort(key=len, reverse=True)
```

&emsp;As componentes são ordenadas pelo tamanho.

```python
if manter_maiores_componentes is not None and manter_maiores_componentes > 0:
    componentes_validos = componentes_validos[:manter_maiores_componentes]
```

&emsp;Opcionalmente, mantém-se apenas as maiores.

```python
filtrada = np.zeros_like(bordas, dtype=np.uint8)
for componente in componentes_validos:
    for linha, coluna in componente:
        filtrada[linha, coluna] = 1
```

&emsp;O resultado final é uma nova imagem binária contendo só os pixels das componentes aprovadas.

### `_rotular_componentes(...)`

&emsp;Essa função implementa rotulagem 8-conectada com busca em largura.

```python
altura, largura = bordas.shape
visitados = np.zeros_like(bordas, dtype=bool)
componentes: List[List[Pixel]] = []
```

Variáveis:

- `altura`, `largura`: limites da imagem;
- `visitados`: matriz booleana para não revisitar pixels;
- `componentes`: lista final de componentes.

```python
for linha in range(altura):
    for coluna in range(largura):
        if bordas[linha, coluna] == 0 or visitados[linha, coluna]:
            continue
```

&emsp;Cada pixel ativo ainda não visitado inicia uma nova componente.

```python
fila: List[Pixel] = [(linha, coluna)]
visitados[linha, coluna] = True
componente: List[Pixel] = []
indice = 0
```

Estrutura da BFS:

- `fila`: pixels a explorar;
- `componente`: pixels já pertencentes à componente atual;
- `indice`: ponteiro manual da fila.


```python
while indice < len(fila):
    atual_linha, atual_coluna = fila[indice]
    indice += 1
    componente.append((atual_linha, atual_coluna))
```

&emsp;Esse laço consome a fila até explorar toda a componente.

```python
for delta_linha in (-1, 0, 1):
    for delta_coluna in (-1, 0, 1):
        if delta_linha == 0 and delta_coluna == 0:
            continue
```

&emsp;Isso percorre os oito vizinhos ao redor do pixel atual.


```python
if not (0 <= vizinho_linha < altura and 0 <= vizinho_coluna < largura):
    continue
if bordas[vizinho_linha, vizinho_coluna] == 0:
    continue
if visitados[vizinho_linha, vizinho_coluna]:
    continue
```

Esses testes garantem:

- vizinho dentro da imagem;
- vizinho ativo;
- vizinho ainda não visitado.


```python
visitados[vizinho_linha, vizinho_coluna] = True
fila.append((vizinho_linha, vizinho_coluna))
```

&emsp;Quando o vizinho é válido, ele entra na fila.

### `_finalizar_segmento_varrido(...)`


```python
segmento_reduzido = _reduzir_rastro(
    segmento,
    max(distancia_minima_entre_pontos, 1.0) ** 2,
)
```

&emsp;Antes de aceitar o segmento, o código remove redundância.


```python
if len(segmento_reduzido) < max(2, pontos_minimos_por_segmento):
    return []
```

&emsp;Esse filtro evita segmentos degenerados demais para o desenho.


```python
return [segmento_reduzido]
```

&emsp;A função retorna uma lista porque, no desenho da pipeline, o trecho finalizado pode ou não virar um segmento válido.

### `_reduzir_rastro(...)`


```python
reduzido: List[Pixel] = [segmento[0]]
ul, uc = segmento[0]
```

&emsp;O segmento reduzido sempre começa no primeiro pixel.


```python
for linha, coluna in segmento[1:]:
    dl = linha - ul
    dc = coluna - uc
    if (dl * dl + dc * dc) < limite_reducao2:
        continue
```

&emsp;Se o novo pixel está perto demais do último pixel aceito, ele é descartado.


```python
reduzido.append((linha, coluna))
ul, uc = linha, coluna
```

&emsp;Quando o pixel ultrapassa a distância mínima, ele é mantido.


```python
if reduzido[-1] != segmento[-1]:
    reduzido.append(segmento[-1])
```

&emsp;O último pixel original é forçado no segmento reduzido para preservar a extremidade do traço.

### `_mapear_para_turtlesim(...)`


```python
linhas = [linha for segmento in segmentos_pixels for linha, _ in segmento]
colunas = [coluna for segmento in segmentos_pixels for _, coluna in segmento]
```

&emsp;O código coleta todos os pixels de todos os segmentos.


```python
linha_min = min(linhas)
linha_max = max(linhas)
coluna_min = min(colunas)
coluna_max = max(colunas)
```

&emsp;Esses valores definem a caixa delimitadora útil do desenho.


```python
largura_util = max((coluna_max - coluna_min) + 1, 1)
altura_util = max((linha_max - linha_min) + 1, 1)
```

&emsp;Evita dimensões zero.


```python
escala_x = tamanho_turtlesim / largura_util
escala_y = tamanho_turtlesim / altura_util
escala = min(escala_x, escala_y)
```

&emsp;A escala escolhida preserva proporção, porque usa o menor fator.


```python
arena = 11.0
largura_desenhada = largura_util * escala
altura_desenhada = altura_util * escala
offset_x = (arena - largura_desenhada) / 2.0
offset_y = (arena - altura_desenhada) / 2.0
```

&emsp;O desenho é centralizado na arena lógica do Turtlesim.


```python
tx = ((coluna - coluna_min) * escala) + offset_x
ty = ((linha_max - linha) * escala) + offset_y
```

Transformação final:

- normaliza pela caixa útil;
- aplica escala;
- inverte o eixo vertical;
- adiciona deslocamento.

---

## 3. `draw_node.py`

### Papel do arquivo

&emsp;Esse arquivo implementa o nó ROS 2 responsável por pegar os segmentos calculados pela pipeline e executá-los no Turtlesim.

Ele precisa resolver quatro problemas:

1. carregar o caminho;
2. saber a posição atual da tartaruga;
3. mover a tartaruga até cada ponto;
4. ligar e desligar a caneta nos momentos certos.

### Imports


```python
import math
from typing import List, Sequence, Tuple
```

Uso:

- `math`: distância euclidiana, `atan2`, seno e cosseno;
- `typing`: clareza dos tipos.


```python
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Empty
from turtlesim.msg import Pose
from turtlesim.srv import Kill, SetPen, Spawn, TeleportAbsolute
```

Uso de cada item:

- `rclpy`: runtime ROS 2 em Python;
- `Twist`: mensagem de velocidade linear e angular;
- `Node`: classe base do nó;
- `QoSProfile`: configuração de qualidade de serviço;
- `Empty`: serviço `/reset`;
- `Pose`: estado da tartaruga;
- `Kill`, `Spawn`, `TeleportAbsolute`: reposicionamento;
- `SetPen`: controle da caneta.


```python
from turtle_draw_pkg.vision_pipeline import gerar_caminho_a_partir_da_imagem
```

&emsp;Esse import faz a ponte direta entre a pipeline e o executor.

### Aliases de tipo


```python
Ponto = Tuple[float, float]
Segmento = List[Ponto]
```

Mesma convenção da pipeline:

- `Ponto`: coordenada contínua do mundo;
- `Segmento`: sequência de pontos que deve ser desenhada continuamente.

### Classe `TurtleDrawNode`

&emsp;Toda a lógica do nó está encapsulada nessa classe.

### `__init__`

&emsp;O construtor faz toda a configuração inicial.

#### a. Nome do nó


```python
super().__init__('draw_node')
```

&emsp;Esse nome é o identificador ROS do processo.

#### b. QoS


```python
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    depth=10,
)
```


- `RELIABLE`: prioridade para entrega confiável;
- `VOLATILE`: não reter mensagens antigas;
- `depth=10`: fila com profundidade 10.

#### c. Parâmetros


```python
self.declare_parameter('image_path', 'edges_final.png')
```

&emsp;Define o caminho da imagem a usar.


```python
self.declare_parameter('min_component_size', 20)
self.declare_parameter('max_components', 80)
self.declare_parameter('point_spacing_pixels', 1.0)
```

&emsp;Esses parâmetros são repassados para a pipeline.


```python
self.declare_parameter('distance_tolerance', 0.08)
self.declare_parameter('linear_gain', 3.0)
self.declare_parameter('angular_gain', 8.0)
self.declare_parameter('max_linear_speed', 2.5)
self.declare_parameter('max_angular_speed', 12.0)
```

&emsp;Esses controlam o comportamento do controlador proporcional.


```python
self.declare_parameter('pen_width', 2)
self.declare_parameter('pen_r', 255)
self.declare_parameter('pen_g', 255)
self.declare_parameter('pen_b', 255)
self.declare_parameter('reset_turtlesim_on_start', True)
self.declare_parameter('pen_settle_cycles', 3)
```

Esses controlam:

- aparência da caneta;
- limpeza inicial da arena;
- espera extra após mudar estado da caneta.

#### d. Tópicos e serviços


```python
self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', qos_profile)
```

&emsp;Publica velocidade para a tartaruga.


```python
self.subscriber_ = self.create_subscription(
    Pose, '/turtle1/pose', self._atualizar_pose, 10
)
```

&emsp;Recebe a pose atual.


```python
self.cliente_caneta = self.create_client(SetPen, '/turtle1/set_pen')
self.cliente_reset = self.create_client(Empty, '/reset')
self.cliente_teleporte = self.create_client(TeleportAbsolute, '/turtle1/teleport_absolute')
self.cliente_kill = self.create_client(Kill, '/kill')
self.cliente_spawn = self.create_client(Spawn, '/spawn')
```

Cada cliente tem uma função específica:

- `set_pen`: ligar/desligar caneta;
- `reset`: limpar arena;
- `teleport_absolute`: reposicionar sem caminhar;
- `kill`: remover a `turtle1`;
- `spawn`: recriá-la já no ponto inicial.

#### e. Estado interno


```python
self.pose_atual: Pose | None = None
self.segmentos: List[Segmento] = self._carregar_segmentos()
self.indice_segmento = 0
self.indice_ponto = 0
self.reposicionando = True
```

Significado:

- `pose_atual`: última pose recebida;
- `segmentos`: caminhos extraídos da imagem;
- `indice_segmento`: qual segmento está sendo executado;
- `indice_ponto`: qual ponto dentro do segmento é o alvo atual;
- `reposicionando`: indica se a tartaruga está indo ao início de um segmento com a caneta desligada.


```python
self.caneta_ativa = None
self.ciclos_espera_caneta = 0
self.futuro_caneta = None
self.futuro_teleporte = None
self.segmento_posicionado = False
self.alvo_teleporte: Ponto | None = None
```


- `caneta_ativa`: estado conhecido localmente; começa como `None` porque o estado real no Turtlesim pode ser desconhecido;
- `ciclos_espera_caneta`: pausa extra depois de mudar a caneta;
- `futuro_caneta`: `Future` de chamada assíncrona do serviço `set_pen`;
- `futuro_teleporte`: `Future` de teleporte em andamento;
- `segmento_posicionado`: indica se a tartaruga já está fisicamente no início do segmento;
- `alvo_teleporte`: ponto que a pose precisa alcançar após o teleporte.

#### f. Preparação inicial


```python
self._resetar_turtlesim_se_necessario()
self._preparar_inicio_desenho()
```

&emsp;Primeiro a arena pode ser limpa. Depois a tartaruga é preparada para começar no ponto certo.

#### g. Timer


```python
self.timer = self.create_timer(0.02, self._desenhar_callback)
```

&emsp;O callback principal roda a cada `0.02` s, ou cerca de 50 Hz.

### `_carregar_segmentos(...)`


```python
caminho = self.get_parameter('image_path').value
min_comp = int(self.get_parameter('min_component_size').value)
max_comp = int(self.get_parameter('max_components').value)
espacamento = float(self.get_parameter('point_spacing_pixels').value)
```

&emsp;Lê os parâmetros relevantes.


```python
segmentos, bordas = gerar_caminho_a_partir_da_imagem(...)
```

&emsp;Pede à pipeline que converta a imagem em segmentos.


```python
n_pixels = int(bordas.sum())
n_pontos = sum(len(s) for s in segmentos)
```

&emsp;Essas estatísticas são usadas para log.


```python
if not segmentos:
    raise RuntimeError(...)
```

&emsp;Se não existir caminho válido, o nó falha logo no início. Isso evita comportamento silencioso errado.

### `_resetar_turtlesim_se_necessario(...)`


```python
if not bool(self.get_parameter('reset_turtlesim_on_start').value):
    return
```

&emsp;O reset é opcional via parâmetro.


```python
if not self.cliente_reset.wait_for_service(timeout_sec=1.0):
    self.get_logger().warning(...)
    return
```

&emsp;Primeiro o código verifica se o serviço existe.


```python
futuro = self.cliente_reset.call_async(Empty.Request())
rclpy.spin_until_future_complete(self, futuro, timeout_sec=2.0)
```

&emsp;O serviço é chamado de modo assíncrono, mas o código espera sua conclusão antes de seguir.


```python
if futuro.done() and futuro.exception() is None:
    self.caneta_ativa = None
```

&emsp;Após reset bem-sucedido, o estado local da caneta volta a ser desconhecido.

### `_preparar_inicio_desenho(...)`


```python
self._reposicionar_tartaruga_no_primeiro_ponto()
self._desligar_caneta_sincronamente()
```

Essa função prepara o começo do desenho antes do timer:

- tenta colocar a tartaruga no primeiro ponto do cachorro;
- garante que a caneta esteja desligada antes de qualquer deslocamento residual.

### `_desligar_caneta_sincronamente(...)`


```python
if not self.cliente_caneta.wait_for_service(timeout_sec=1.0):
    ...
```

&emsp;Verifica se o serviço está disponível.


```python
req = SetPen.Request()
req.off = 1
```

&emsp;Monta uma requisição de caneta desligada.


```python
self._parar_movimento()
futuro = self.cliente_caneta.call_async(req)
rclpy.spin_until_future_complete(self, futuro, timeout_sec=2.0)
```

&emsp;Antes de trocar o estado da caneta, o nó publica velocidade zero.


```python
if futuro.done() and futuro.exception() is None:
    self.caneta_ativa = False
```

&emsp;Se tudo der certo, o estado local é sincronizado.

### `_teleportar_para_primeiro_segmento_sincronamente(...)`

&emsp;Essa função é o fallback para posicionar a tartaruga no primeiro ponto sem o fluxo de `kill/spawn`.


```python
seg = self._segmento_atual()
if seg is None:
    return
```

&emsp;Sem segmento, não há para onde ir.


```python
req = TeleportAbsolute.Request()
req.x = float(ponto_inicial[0])
req.y = float(ponto_inicial[1])
req.theta = 0.0
```

&emsp;Define o teleporte para o primeiro ponto do primeiro segmento.


```python
self._parar_movimento()
futuro = self.cliente_teleporte.call_async(req)
rclpy.spin_until_future_complete(self, futuro, timeout_sec=2.0)
```

&emsp;O nó para o movimento e espera o teleporte terminar.

### `_reposicionar_tartaruga_no_primeiro_ponto(...)`

&emsp;Essa função tenta evitar a linha inicial causada pela `turtle1` nascer no centro da arena.


```python
if not self.cliente_kill.wait_for_service(timeout_sec=1.0):
    self._teleportar_para_primeiro_segmento_sincronamente()
    return
```

&emsp;Se `kill` ou `spawn` não existirem, o código cai no teleporte.


```python
req_kill = Kill.Request()
req_kill.name = 'turtle1'
```

&emsp;Primeiro a tartaruga original é removida.


```python
req_spawn = Spawn.Request()
req_spawn.x = float(ponto_inicial[0])
req_spawn.y = float(ponto_inicial[1])
req_spawn.theta = 0.0
req_spawn.name = 'turtle1'
```

&emsp;Depois ela é recriada já no ponto inicial do desenho.


```python
if futuro_spawn.done() and futuro_spawn.exception() is None:
    self.segmento_posicionado = True
```

&emsp;Se o spawn funcionar, o primeiro segmento já nasce posicionado.

### `_atualizar_pose(...)`


```python
self.pose_atual = msg
```

&emsp;É um callback simples: sempre guarda a última pose recebida.

### `_atualizar_caneta(...)`


```python
if self.caneta_ativa == ativa:
    return
```

&emsp;Evita chamada desnecessária se o estado já for o desejado.


```python
req = SetPen.Request()
req.off = 0 if ativa else 1
```

&emsp;Converte o estado lógico em requisição do serviço.


```python
self.publisher_.publish(Twist())
self.futuro_caneta = self.cliente_caneta.call_async(req)
self.caneta_ativa = ativa
self.ciclos_espera_caneta = int(self.get_parameter('pen_settle_cycles').value)
```

O nó:

- zera o movimento;
- dispara a requisição assíncrona;
- registra o estado esperado;
- impõe alguns ciclos extras de espera.


- foi uma defesa adicionada após vários traços artificiais entre segmentos.

### `_parar_movimento(...)`


```python
self.publisher_.publish(Twist())
```

&emsp;Função simples, mas crucial. Ela publica velocidade nula para eliminar movimento residual antes de trocar caneta, teleportar ou finalizar.

### `_segmento_atual(...)`


```python
if self.indice_segmento >= len(self.segmentos):
    return None
return self.segmentos[self.indice_segmento]
```

&emsp;Retorna o segmento corrente ou `None` se o desenho acabou.

### `_alvo_atual(...)`


```python
seg = self._segmento_atual()
if seg is None or self.indice_ponto >= len(seg):
    return None
return seg[self.indice_ponto]
```

&emsp;Retorna o ponto atual dentro do segmento corrente.

### `_avancar_segmento(...)`


```python
self._parar_movimento()
self.indice_segmento += 1
self.indice_ponto = 0
self.reposicionando = True
self.segmento_posicionado = False
self._atualizar_caneta(ativa=False)
```

Quando um segmento acaba:

- o nó para a tartaruga;
- vai para o próximo segmento;
- reinicia o índice de ponto;
- entra em modo de reposicionamento;
- marca que ainda não está no início do novo segmento;
- desliga a caneta.

&emsp;O retorno booleano indica se ainda há segmentos restantes.

### `_finalizar(...)`


```python
self._parar_movimento()
self._atualizar_caneta(ativa=False)
self.timer.cancel()
```

Ao final:

- para tudo;
- desliga caneta;
- cancela o loop principal.

### `_desenhar_callback(...)`

&emsp;Essa é a função central do nó. Ela implementa o autômato de estados do desenho.

#### a. Espera pela pose


```python
if self.pose_atual is None:
    return
```

&emsp;Sem pose, o controlador não consegue operar.

#### b. Espera por teleporte assíncrono


```python
if self.futuro_teleporte is not None:
    if not self.futuro_teleporte.done():
        self.publisher_.publish(Twist())
        return
    self.futuro_teleporte = None
    return
```

Se há teleporte em andamento:

- o nó não move a tartaruga;
- publica `Twist()` zero;
- só continua quando o serviço termina.

#### c. Confirmação por pose após teleporte


```python
if self.reposicionando and self.alvo_teleporte is not None:
    erro_x = self.alvo_teleporte[0] - self.pose_atual.x
    erro_y = self.alvo_teleporte[1] - self.pose_atual.y
```

&emsp;Mesmo com o `Future` concluído, o código ainda espera a pose refletir o ponto esperado.


```python
if math.hypot(erro_x, erro_y) >= float(self.get_parameter('distance_tolerance').value):
    self.publisher_.publish(Twist())
    return
```

&emsp;Enquanto a pose não coincidir com o alvo do teleporte, nada anda.


```python
self.segmento_posicionado = True
self.alvo_teleporte = None
```

&emsp;Quando a pose confirma o teleporte, o segmento passa a ser considerado posicionado.

#### d. Espera por troca de caneta


```python
if self.futuro_caneta is not None:
    if not self.futuro_caneta.done():
        self.publisher_.publish(Twist())
        return
    self.futuro_caneta = None
```

&emsp;Enquanto a caneta estiver trocando de estado, o movimento fica bloqueado.

#### e. Espera extra de assentamento


```python
if self.ciclos_espera_caneta > 0:
    self.publisher_.publish(Twist())
    self.ciclos_espera_caneta -= 1
    return
```

&emsp;Essa espera foi introduzida para evitar risco residual logo após `set_pen`.

#### f. Reposicionamento ao início do segmento


```python
if self.reposicionando and not self.segmento_posicionado:
    if not self._teleportar_para_inicio_do_segmento():
        self._finalizar()
    return
```

&emsp;Se o nó ainda não está no início do segmento, ele tenta teleportar para lá com a caneta desligada.

#### g. Leitura do alvo atual


```python
alvo = self._alvo_atual()
if alvo is None:
    self._finalizar()
    return
```

&emsp;Sem alvo, o desenho termina.

#### h. Cálculo do erro geométrico


```python
erro_x = alvo_x - self.pose_atual.x
erro_y = alvo_y - self.pose_atual.y
distancia = math.hypot(erro_x, erro_y)
```

&emsp;Aqui o nó mede a distância ao ponto alvo.


```python
angulo_alvo = math.atan2(erro_y, erro_x)
erro_angular = angulo_alvo - self.pose_atual.theta
erro_angular = math.atan2(math.sin(erro_angular), math.cos(erro_angular))
```

&emsp;O erro angular é normalizado para `[-pi, pi]`.

#### i. Verificação de tolerância


```python
if distancia < tolerancia:
    self._ponto_atingido()
    return
```

&emsp;Se o ponto foi alcançado, o estado avança.

#### j. Controle contínuo


```python
self._publicar_velocidade(distancia, erro_angular)
```

&emsp;Caso contrário, o nó calcula e publica um comando de movimento.

### `_ponto_atingido(...)`

&emsp;Essa função trata a transição de estado quando a tartaruga chega ao alvo atual.

#### a. Caso reposicionando


```python
if self.reposicionando:
    seg = self._segmento_atual()
```

Se a tartaruga estava só se posicionando no início do segmento:

- ainda não começou a desenhar aquele segmento;
- chegou apenas no ponto inicial.


```python
if len(seg) <= 1:
    if not self._avancar_segmento():
        self._finalizar()
    return
```

&emsp;Segmentos degenerados são pulados.


```python
self.reposicionando = False
self._parar_movimento()
self._atualizar_caneta(ativa=True)
self.indice_ponto = 1
```

Quando o início do segmento foi alcançado:

- sai do modo de reposicionamento;
- para movimento;
- liga a caneta;
- define o próximo alvo como o segundo ponto, porque o primeiro já foi alcançado.

#### b. Caso desenhando


```python
self.indice_ponto += 1
```

&emsp;Se a tartaruga já estava desenhando, apenas avança para o próximo ponto.


```python
if seg is not None and self.indice_ponto < len(seg):
    return
```

&emsp;Se ainda há pontos no segmento, o callback seguinte continuará normalmente.


```python
if not self._avancar_segmento():
    self._finalizar()
```

&emsp;Se acabou o segmento, o nó passa ao próximo. Se não houver mais segmentos, encerra.

### `_publicar_velocidade(...)`

&emsp;Esse é o controlador proporcional do movimento.


```python
g_lin = float(self.get_parameter('linear_gain').value)
g_ang = float(self.get_parameter('angular_gain').value)
v_lin_max = float(self.get_parameter('max_linear_speed').value)
v_ang_max = float(self.get_parameter('max_angular_speed').value)
```

&emsp;O ganho e os limites são parametrizados.


```python
cmd = Twist()
cmd.linear.x = min(g_lin * distancia, v_lin_max)
cmd.angular.z = max(-v_ang_max, min(g_ang * erro_angular, v_ang_max))
```

Lógica:

- velocidade linear proporcional à distância;
- velocidade angular proporcional ao erro angular;
- ambas saturadas.


```python
if abs(erro_angular) > 0.6:
    cmd.linear.x = 0.0
elif abs(erro_angular) > 0.25:
    cmd.linear.x *= 0.15
```

&emsp;Essas duas regras impedem que a tartaruga avance forte demais enquanto ainda está mal orientada.


```python
self.publisher_.publish(cmd)
```

&emsp;O comando é então publicado.

### `_teleportar_para_inicio_do_segmento(...)`

&emsp;Essa função faz o reposicionamento entre segmentos.


```python
seg = self._segmento_atual()
if seg is None:
    return False
```

&emsp;Sem segmento, nada a fazer.


```python
req = TeleportAbsolute.Request()
req.x = float(ponto_inicial[0])
req.y = float(ponto_inicial[1])
req.theta = 0.0
```

&emsp;O alvo do teleporte é o primeiro ponto do segmento atual.


```python
self._parar_movimento()
self.segmento_posicionado = False
self.alvo_teleporte = ponto_inicial
self.futuro_teleporte = self.cliente_teleporte.call_async(req)
```

O código:

- zera o movimento;
- marca que o segmento ainda não está posicionado;
- guarda o alvo esperado;
- dispara o teleporte assíncrono.

&emsp;O callback principal só continua quando o `Future` acabar e a pose confirmar o ponto.

### `main(...)`


```python
rclpy.init(args=args)
node = TurtleDrawNode()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()
```

Fluxo:

1. inicializa o ROS;
2. cria o nó;
3. mantém o processo executando;
4. destrói o nó ao final;
5. encerra o ROS.


