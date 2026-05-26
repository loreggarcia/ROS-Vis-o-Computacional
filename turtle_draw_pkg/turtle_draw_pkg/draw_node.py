"""No ROS 2 para desenhar os contornos de uma imagem no turtlesim.

    1. Carregar segmentos pre-calculados pela vision_pipeline;
    2. Receber a pose atual da tartaruga;
    3. Aplicar controle proporcional P;
    4. Ligar e desligar a caneta entre segmentos desconectados.

    1. No notebook, exporte a imagem final de bordas:
           import cv2
           cv2.imwrite('edges_final.png', bordas_binarias * 255)
    2. Passe o caminho do PNG via parametro image_path:
           ros2 run turtle_draw_pkg draw_node --ros-args -p image_path:=/caminho/edges_final.png
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Empty
from turtlesim.msg import Pose
from turtlesim.srv import Kill, SetPen, Spawn, TeleportAbsolute

from turtle_draw_pkg.vision_pipeline import gerar_caminho_a_partir_da_imagem

Ponto = Tuple[float, float]
Segmento = List[Ponto]


class TurtleDrawNode(Node):
    """No responsavel por desenhar contornos de imagem no Turtlesim."""

    def __init__(self):
        super().__init__('draw_node')

        # QoS basico e suficiente para comandos de velocidade e pose no turtlesim.
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # --- Parametros ---
        self.declare_parameter('image_path', 'edges_final.png')
        # Filtragem de contornos
        self.declare_parameter('min_component_size', 2)
        self.declare_parameter('max_components', 500)
        self.declare_parameter('point_spacing_pixels', 1.0)
        # Controle
        self.declare_parameter('distance_tolerance', 0.08)
        self.declare_parameter('linear_gain', 3.0)
        self.declare_parameter('angular_gain', 8.0)
        self.declare_parameter('max_linear_speed', 2.5)
        self.declare_parameter('max_angular_speed', 12.0)
        # Caneta
        self.declare_parameter('pen_width', 2)
        self.declare_parameter('pen_r', 255)
        self.declare_parameter('pen_g', 255)
        self.declare_parameter('pen_b', 255)
        self.declare_parameter('reset_turtlesim_on_start', True)
        self.declare_parameter('pen_settle_cycles', 3)

        # --- Topicos e servicos ---
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', qos_profile)
        self.subscriber_ = self.create_subscription(
            Pose, '/turtle1/pose', self._atualizar_pose, 10
        )
        self.cliente_caneta = self.create_client(SetPen, '/turtle1/set_pen')
        self.cliente_reset = self.create_client(Empty, '/reset')
        self.cliente_teleporte = self.create_client(
            TeleportAbsolute,
            '/turtle1/teleport_absolute',
        )
        self.cliente_kill = self.create_client(Kill, '/kill')
        self.cliente_spawn = self.create_client(Spawn, '/spawn')

        # --- Estado ---
        self.pose_atual: Pose | None = None
        # A pipeline ja entrega o desenho quebrado em segmentos e pontos.
        self.segmentos: List[Segmento] = self._carregar_segmentos()
        self.indice_segmento = 0
        self.indice_ponto = 0
        # Quando True, a tartaruga esta apenas se reposicionando, sem desenhar.
        self.reposicionando = True
        # No turtlesim a caneta costuma iniciar ligada; portanto o estado real
        # aqui e desconhecido ate sincronizarmos via servico.
        self.caneta_ativa = None
        self.ciclos_espera_caneta = 0
        self.futuro_caneta = None
        self.futuro_teleporte = None
        self.segmento_posicionado = False
        self.alvo_teleporte: Ponto | None = None

        self._resetar_turtlesim_se_necessario()

        self._preparar_inicio_desenho()

        self.timer = self.create_timer(0.02, self._desenhar_callback)
        self.get_logger().info(
            f'Imagem carregada: {len(self.segmentos)} segmento(s). '
            f'Total de pontos: {sum(len(s) for s in self.segmentos)}. '
            'Iniciando desenho no Turtlesim.'
        )

    # ------------------------------------------------------------------
    # Inicializacao
    # ------------------------------------------------------------------

    def _carregar_segmentos(self) -> List[Segmento]:
        """Carrega a imagem de bordas e converte em segmentos."""

        caminho = self.get_parameter('image_path').value
        min_comp = int(self.get_parameter('min_component_size').value)
        max_comp = int(self.get_parameter('max_components').value)
        espacamento = float(self.get_parameter('point_spacing_pixels').value)

        self.get_logger().info(f'Carregando imagem de bordas: {caminho}')

        # A vision_pipeline converte os pixels de borda em trajetorias prontas para desenho.
        segmentos, bordas = gerar_caminho_a_partir_da_imagem(
            caminho_imagem=caminho,
            tamanho_minimo_componente=min_comp,
            manter_maiores_componentes=max_comp,
            distancia_minima_entre_pontos=espacamento,
        )

        n_pixels = int(bordas.sum())
        n_pontos = sum(len(s) for s in segmentos)
        self.get_logger().info(
            f'Pixels de borda: {n_pixels}. '
            f'Segmentos gerados: {len(segmentos)}. '
            f'Pontos apos reducao: {n_pontos}.'
        )

        if not segmentos:
            raise RuntimeError(
                'Nenhum segmento encontrado. '
                'Verifique se a imagem contem pixels brancos sobre fundo preto '
                'e ajuste min_component_size se necessario.'
            )

        return segmentos

    def _resetar_turtlesim_se_necessario(self) -> None:
        """Limpa os rastros antigos e recoloca a tartaruga no estado inicial."""

        if not bool(self.get_parameter('reset_turtlesim_on_start').value):
            return

        if not self.cliente_reset.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning('Servico /reset indisponivel; seguindo sem limpar a tela.')
            return

        futuro = self.cliente_reset.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, futuro, timeout_sec=2.0)
        if futuro.done() and futuro.exception() is None:
            self.caneta_ativa = None
            self.get_logger().info('Turtlesim resetado antes do desenho.')
            return

        self.get_logger().warning('Nao foi possivel confirmar o reset do Turtlesim.')

    def _preparar_inicio_desenho(self) -> None:
        """Desliga a caneta e posiciona no primeiro segmento antes do timer."""

        # O desenho sempre comeca ja no primeiro ponto util para evitar rastro inicial.
        self._reposicionar_tartaruga_no_primeiro_ponto()
        self._desligar_caneta_sincronamente()

    def _desligar_caneta_sincronamente(self) -> None:
        """Garante caneta desligada antes de qualquer movimento inicial."""

        if not self.cliente_caneta.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning('Servico /turtle1/set_pen indisponivel no inicio.')
            return

        req = SetPen.Request()
        req.r = int(self.get_parameter('pen_r').value)
        req.g = int(self.get_parameter('pen_g').value)
        req.b = int(self.get_parameter('pen_b').value)
        req.width = int(self.get_parameter('pen_width').value)
        req.off = 1
        self._parar_movimento()
        futuro = self.cliente_caneta.call_async(req)
        rclpy.spin_until_future_complete(self, futuro, timeout_sec=2.0)
        if futuro.done() and futuro.exception() is None:
            self.caneta_ativa = False
            self.ciclos_espera_caneta = 0
            self.futuro_caneta = None
            return

        self.get_logger().warning('Nao foi possivel confirmar caneta desligada no inicio.')

    def _teleportar_para_primeiro_segmento_sincronamente(self) -> None:
        """Posiciona a tartaruga no primeiro ponto antes do loop principal."""

        seg = self._segmento_atual()
        if seg is None:
            return

        if not self.cliente_teleporte.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning('Servico /turtle1/teleport_absolute indisponivel no inicio.')
            return

        ponto_inicial = seg[0]
        req = TeleportAbsolute.Request()
        req.x = float(ponto_inicial[0])
        req.y = float(ponto_inicial[1])
        req.theta = 0.0
        self._parar_movimento()
        futuro = self.cliente_teleporte.call_async(req)
        rclpy.spin_until_future_complete(self, futuro, timeout_sec=2.0)
        if futuro.done() and futuro.exception() is None:
            self.segmento_posicionado = True
            self.alvo_teleporte = None
            self.futuro_teleporte = None
            return

        self.get_logger().warning('Nao foi possivel confirmar teleporte inicial.')

    def _reposicionar_tartaruga_no_primeiro_ponto(self) -> None:
        """Substitui a turtle inicial por uma nova ja no primeiro ponto do desenho."""

        seg = self._segmento_atual()
        if seg is None:
            return

        ponto_inicial = seg[0]

        # A estrategia preferida e recriar a turtle no ponto inicial.
        # Se isso falhar, o codigo recua para teleporte.
        if not self.cliente_kill.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning('Servico /kill indisponivel; usando teleporte inicial.')
            self._teleportar_para_primeiro_segmento_sincronamente()
            return

        if not self.cliente_spawn.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning('Servico /spawn indisponivel; usando teleporte inicial.')
            self._teleportar_para_primeiro_segmento_sincronamente()
            return

        req_kill = Kill.Request()
        req_kill.name = 'turtle1'
        futuro_kill = self.cliente_kill.call_async(req_kill)
        rclpy.spin_until_future_complete(self, futuro_kill, timeout_sec=2.0)
        if not futuro_kill.done() or futuro_kill.exception() is not None:
            self.get_logger().warning('Nao foi possivel remover turtle1; usando teleporte inicial.')
            self._teleportar_para_primeiro_segmento_sincronamente()
            return

        req_spawn = Spawn.Request()
        req_spawn.x = float(ponto_inicial[0])
        req_spawn.y = float(ponto_inicial[1])
        req_spawn.theta = 0.0
        req_spawn.name = 'turtle1'
        futuro_spawn = self.cliente_spawn.call_async(req_spawn)
        rclpy.spin_until_future_complete(self, futuro_spawn, timeout_sec=2.0)
        if futuro_spawn.done() and futuro_spawn.exception() is None:
            self.segmento_posicionado = True
            self.alvo_teleporte = None
            self.futuro_teleporte = None
            return

        self.get_logger().warning('Nao foi possivel recriar turtle1 no ponto inicial; usando teleporte inicial.')
        self._teleportar_para_primeiro_segmento_sincronamente()

    # ------------------------------------------------------------------
    # Callbacks e helpers de pose
    # ------------------------------------------------------------------

    def _atualizar_pose(self, msg: Pose) -> None:
        # Guarda a pose mais recente recebida do turtlesim.
        self.pose_atual = msg

    # ------------------------------------------------------------------
    # Controle de caneta
    # ------------------------------------------------------------------

    def _atualizar_caneta(self, ativa: bool) -> bool:
        """Liga ou desliga a caneta; retorna True se o pedido foi aceito."""

        if self.caneta_ativa == ativa:
            return True

        # Apos respawn/teleporte o servico pode levar alguns instantes para reaparecer.
        if not self.cliente_caneta.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning('Servico /turtle1/set_pen indisponivel.')
            return False

        req = SetPen.Request()
        req.r = int(self.get_parameter('pen_r').value)
        req.g = int(self.get_parameter('pen_g').value)
        req.b = int(self.get_parameter('pen_b').value)
        req.width = int(self.get_parameter('pen_width').value)
        req.off = 0 if ativa else 1
        # Para antes de trocar o estado da caneta para evitar pequenos riscos.
        self.publisher_.publish(Twist())
        self.futuro_caneta = self.cliente_caneta.call_async(req)
        self.caneta_ativa = ativa
        self.ciclos_espera_caneta = int(self.get_parameter('pen_settle_cycles').value)
        return True

    def _parar_movimento(self) -> None:
        """Publica um comando nulo para evitar risco residual entre estados."""

        self.publisher_.publish(Twist())

    # ------------------------------------------------------------------
    # Navegacao entre segmentos
    # ------------------------------------------------------------------

    def _segmento_atual(self) -> Sequence[Ponto] | None:
        if self.indice_segmento >= len(self.segmentos):
            return None
        return self.segmentos[self.indice_segmento]

    def _alvo_atual(self) -> Ponto | None:
        seg = self._segmento_atual()
        if seg is None or self.indice_ponto >= len(seg):
            return None
        return seg[self.indice_ponto]

    def _avancar_segmento(self) -> bool:
        """Vai para o proximo segmento. Retorna False se acabou."""
        # Ao trocar de segmento, a caneta desliga para nao criar linhas artificiais.
        self._parar_movimento()
        self.indice_segmento += 1
        self.indice_ponto = 0
        self.reposicionando = True
        self.segmento_posicionado = False
        self._atualizar_caneta(ativa=False)
        return self.indice_segmento < len(self.segmentos)

    def _finalizar(self) -> None:
        self._parar_movimento()
        self._atualizar_caneta(ativa=False)
        self.get_logger().info('Desenho concluido!')
        self.timer.cancel()

    # ------------------------------------------------------------------
    # Loop principal de controle
    # ------------------------------------------------------------------

    def _desenhar_callback(self) -> None:
        # Esse timer e a maquina de estados principal do no.
        if self.pose_atual is None:
            return

        if self.futuro_teleporte is not None:
            # Enquanto um teleporte nao termina, nao publica movimento.
            if not self.futuro_teleporte.done():
                self.publisher_.publish(Twist())
                return
            self.futuro_teleporte = None
            return

        if self.reposicionando and self.alvo_teleporte is not None:
            if self.pose_atual is None:
                return
            # Mesmo apos o servico retornar, confirma se a pose realmente chegou ao alvo.
            erro_x = self.alvo_teleporte[0] - self.pose_atual.x
            erro_y = self.alvo_teleporte[1] - self.pose_atual.y
            if math.hypot(erro_x, erro_y) >= float(self.get_parameter('distance_tolerance').value):
                self.publisher_.publish(Twist())
                return
            self.segmento_posicionado = True
            self.alvo_teleporte = None

        # Aguarda o servico da caneta confirmar a troca de estado.
        if self.futuro_caneta is not None:
            if not self.futuro_caneta.done():
                self.publisher_.publish(Twist())
                return
            self.futuro_caneta = None

        if self.ciclos_espera_caneta > 0:
            # Aguarda alguns ciclos para o turtlesim estabilizar a troca da caneta.
            self.publisher_.publish(Twist())
            self.ciclos_espera_caneta -= 1
            return

        if self.reposicionando and not self.segmento_posicionado:
            # Se entrou num novo segmento, primeiro se reposiciona com a caneta desligada.
            if not self._teleportar_para_inicio_do_segmento():
                self._finalizar()
            return

        alvo = self._alvo_atual()
        if alvo is None:
            self._finalizar()
            return

        alvo_x, alvo_y = alvo
        erro_x = alvo_x - self.pose_atual.x
        erro_y = alvo_y - self.pose_atual.y
        distancia = math.hypot(erro_x, erro_y)

        # Calcula a direcao desejada e compara com o angulo atual da tartaruga.
        angulo_alvo = math.atan2(erro_y, erro_x)
        erro_angular = angulo_alvo - self.pose_atual.theta
        # Normaliza para [-pi, pi]
        erro_angular = math.atan2(math.sin(erro_angular), math.cos(erro_angular))

        tolerancia = float(self.get_parameter('distance_tolerance').value)

        if distancia < tolerancia:
            self._ponto_atingido()
            return

        self._publicar_velocidade(distancia, erro_angular)

    def _ponto_atingido(self) -> None:
        """Logica executada quando a tartaruga chegou ao ponto atual."""

        if self.reposicionando:
            seg = self._segmento_atual()
            if seg is None:
                self._finalizar()
                return

            if len(seg) <= 1:
                # Segmento degenerado, pula
                if not self._avancar_segmento():
                    self._finalizar()
                return

            # So entra no modo de desenho depois de conseguir solicitar a caneta ligada.
            self._parar_movimento()
            if not self._atualizar_caneta(ativa=True):
                return
            self.reposicionando = False
            self.indice_ponto = 1  # ja chegou ao ponto 0; proximo alvo e 1
            return

        # Avanca para o proximo ponto dentro do segmento
        self.indice_ponto += 1
        seg = self._segmento_atual()
        if seg is not None and self.indice_ponto < len(seg):
            return  # ainda ha pontos neste segmento

        # Fim do segmento: vai para o proximo
        if not self._avancar_segmento():
            self._finalizar()

    def _publicar_velocidade(self, distancia: float, erro_angular: float) -> None:
        """Calcula e publica o comando Twist com controle P."""

        g_lin = float(self.get_parameter('linear_gain').value)
        g_ang = float(self.get_parameter('angular_gain').value)
        v_lin_max = float(self.get_parameter('max_linear_speed').value)
        v_ang_max = float(self.get_parameter('max_angular_speed').value)

        cmd = Twist()
        # Controle proporcional: quanto maior o erro, maior o comando, ate o limite maximo.
        cmd.linear.x = min(g_lin * distancia, v_lin_max)
        cmd.angular.z = max(-v_ang_max, min(g_ang * erro_angular, v_ang_max))

        # Nao translada quando a orientacao ainda esta muito errada.
        if abs(erro_angular) > 0.6:
            cmd.linear.x = 0.0
        elif abs(erro_angular) > 0.25:
            cmd.linear.x *= 0.15

        self.publisher_.publish(cmd)

    def _teleportar_para_inicio_do_segmento(self) -> bool:
        """Reposiciona no inicio do segmento com a caneta desligada."""

        seg = self._segmento_atual()
        if seg is None:
            return False

        ponto_inicial = seg[0]
        if not self.cliente_teleporte.wait_for_service(timeout_sec=0.1):
            self.get_logger().warning('Servico /turtle1/teleport_absolute indisponivel.')
            return False

        req = TeleportAbsolute.Request()
        req.x = float(ponto_inicial[0])
        req.y = float(ponto_inicial[1])
        req.theta = 0.0
        self._parar_movimento()
        # Marca que ainda estamos aguardando a confirmacao desse reposicionamento.
        self.segmento_posicionado = False
        self.alvo_teleporte = ponto_inicial
        self.futuro_teleporte = self.cliente_teleporte.call_async(req)
        return True


def main(args=None):
    # Inicializa o ROS, cria o no e mantem o loop ativo ate encerrar.
    rclpy.init(args=args)
    node = TurtleDrawNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
