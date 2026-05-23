# Turtle Draw ROS

Projeto em ROS 2 para desenhar, no `turtlesim`, o contorno de uma imagem processada no notebook.

## Visão geral

O fluxo do projeto é dividido em três etapas:

1. `processamento.ipynb` processa `dog.jpg` e gera a imagem final de bordas.
2. `turtle_draw_pkg/turtle_draw_pkg/vision_pipeline.py` lê `edges_final.png` e converte os pixels ativos em segmentos no espaço lógico do `turtlesim`.
3. `turtle_draw_pkg/turtle_draw_pkg/draw_node.py` percorre esses segmentos no ROS 2, controlando movimento, teleporte e caneta.

## Estrutura das pastas

```text
.
├── docs/
│   └── relatorio_arquitetura.md
├── processamento.ipynb
├── dog.jpg
├── edges_final.png
├── README.md
└── turtle_draw_pkg/
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── resource/
    │   └── turtle_draw_pkg
    ├── test/
    │   ├── test_copyright.py
    │   ├── test_flake8.py
    │   └── test_pep257.py
    └── turtle_draw_pkg/
        ├── __init__.py
        ├── draw_node.py
        └── vision_pipeline.py
```

## O que cada parte faz

- `processamento.ipynb`: concentra as etapas de visão computacional e a validação visual do contorno.
- `dog.jpg`: imagem de entrada usada no notebook.
- `edges_final.png`: saída final do notebook, usada como entrada do runtime ROS.
- `docs/relatorio_arquitetura.md`: relatório explicando o código e as decisões de implementação.
- `turtle_draw_pkg/package.xml`: metadados e dependências ROS 2 do pacote.
- `turtle_draw_pkg/setup.py` e `setup.cfg`: configuração do pacote Python.
- `turtle_draw_pkg/turtle_draw_pkg/vision_pipeline.py`: converte a imagem binária em segmentos para o `turtlesim`.
- `turtle_draw_pkg/turtle_draw_pkg/draw_node.py`: nó ROS 2 que desenha os segmentos.
- `turtle_draw_pkg/test/`: testes básicos de estilo e empacotamento.

## Pré-requisitos

- ROS 2 Humble
- `turtlesim`
- Python com `numpy` e `opencv-python`
- ambiente onde o comando `ros2` esteja disponível

Neste workspace, os comandos foram executados com o ambiente:

```bash
export PATH="/home/lorena/.local/share/mamba/envs/ros2-humble/bin:$PATH"
```

## Como gerar a imagem de bordas

1. Abra `processamento.ipynb`.
2. Execute as células do notebook até obter `bordas_binarias`.
3. Exporte a imagem final:

```python
import cv2
cv2.imwrite('edges_final.png', bordas_binarias * 255)
```

Se `edges_final.png` já existir e estiver correto, essa etapa pode ser pulada.

## Como compilar

Na raiz do projeto:

```bash
/home/lorena/.local/share/mamba/envs/ros2-humble/bin/colcon build --packages-select turtle_draw_pkg
source install/setup.bash
```

## Como executar

Em um terminal, inicie o `turtlesim`:

```bash
export PATH="/home/lorena/.local/share/mamba/envs/ros2-humble/bin:$PATH"
cd /home/lorena/ponderada-ROS
source install/setup.bash
ros2 run turtlesim turtlesim_node
```

Em outro terminal, execute o nó de desenho:

```bash
export PATH="/home/lorena/.local/share/mamba/envs/ros2-humble/bin:$PATH"
cd /home/lorena/ponderada-ROS
source install/setup.bash
ros2 run turtle_draw_pkg draw_node --ros-args -p image_path:=/home/lorena/ponderada-ROS/edges_final.png
```

## Parâmetros úteis do nó

Os parâmetros mais relevantes de `draw_node.py` são:

- `image_path`: caminho do PNG binário final.
- `min_component_size`: tamanho mínimo de componente conectada mantida.
- `max_components`: número máximo de componentes preservadas.
- `point_spacing_pixels`: espaçamento mínimo entre pontos após redução.
- `distance_tolerance`: tolerância para considerar um ponto atingido.
- `linear_gain` e `angular_gain`: ganhos do controlador proporcional.
- `max_linear_speed` e `max_angular_speed`: saturações de velocidade.
- `pen_width`: largura da caneta.
- `reset_turtlesim_on_start`: limpa a arena antes de desenhar.

Exemplo com parâmetros explícitos:

```bash
ros2 run turtle_draw_pkg draw_node --ros-args \
  -p image_path:=/home/lorena/ponderada-ROS/edges_final.png \
  -p min_component_size:=20 \
  -p max_components:=80 \
  -p point_spacing_pixels:=1.0
```

## Observações

- `build/`, `install/`, `log/` e `__pycache__/` são artefatos gerados e não fazem parte do código-fonte.
- Sempre que alterar o pacote ROS, faça novo `colcon build`.
- Se `ros2` não for encontrado, o problema é ambiente, não o código do projeto.
