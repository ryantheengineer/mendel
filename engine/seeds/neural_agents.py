# engine/seeds/neural_agents.py

import random
import copy
from ..seed import MendelSeed
from ..agents.neural_agent import NeuralAgent
from ..canvas.canvas_environment import CanvasEnvironment
from simplification.cutil import simplify_coords

class NeuralAgentSeed(MendelSeed):

    def __init__(self, config):
        # Canvas setup
        self.canvas = CanvasEnvironment(config.get('canvas', {}))

        # Network shape
        self.network_config = config.get('network', {
            'input_size': 6,    # x, y, edge_dist, density, attr_dist, attr_angle
            'hidden_size': 8,
            'output_size': 2    # turn_rate, speed
        })

        # Agent behavior caps
        self.n_agents = config.get('n_agents', 20)
        self.step_scale = config.get('step_scale', 3.0)
        self.turn_scale = config.get('turn_scale', 0.3)

    @property
    def name(self):
        return "Neural Agent"

    @property
    def canvas_width(self):
        return self.canvas.width

    @property
    def canvas_height(self):
        return self.canvas.height

    def _n_weights(self):
        nc = self.network_config
        return nc['input_size'] * nc['hidden_size'] + \
               nc['hidden_size'] * nc['output_size']

    def random_genome(self):
        return {
            'weights': [random.uniform(-2.0, 2.0)
                       for _ in range(self._n_weights())],
            'lifespan': random.randint(100, 5000),
            'step_scale': random.uniform(1.0, 5.0),
            'turn_scale': random.uniform(0.1, 1.0),
        }

    def mutate(self, genome, rate=0.15):
        g = copy.deepcopy(genome)
        g['weights'] = [
            w + random.gauss(0, rate) if random.random() < 0.3 else w
            for w in g['weights']
        ]
        if random.random() < 0.2:
            g['lifespan'] = max(100, g['lifespan'] + random.randint(-200, 200))
        return g

    def crossover(self, genome_a, genome_b):
        cut = random.randint(0, len(genome_a['weights']))
        return {
            'weights': genome_a['weights'][:cut] + genome_b['weights'][cut:],
            'lifespan': random.choice([genome_a['lifespan'],
                                       genome_b['lifespan']]),
            'step_scale': (genome_a['step_scale'] +
                           genome_b['step_scale']) / 2,
            'turn_scale': random.choice([genome_a['turn_scale'],
                                         genome_b['turn_scale']]),
        }

    def _sense(self, agent, canvas):
        """Gather normalized inputs for the agent's network."""
        nx = agent.x / canvas.width
        ny = agent.y / canvas.height
        edge = canvas.edge_distance(agent.x, agent.y)
        density = canvas.sample_density(agent.x, agent.y)
        attr_dist, attr_angle, _, _ = canvas.attractor_inputs(
            agent.x, agent.y
        )
        return [nx, ny, edge, density, attr_dist, attr_angle]
    
    def render(self, genome, output_path):
        self.canvas.reset()

        agents = [
            NeuralAgent(
                x=random.uniform(0.1, 0.9) * self.canvas.width,
                y=random.uniform(0.1, 0.9) * self.canvas.height,
                heading=random.uniform(0, 6.28),
                network_config=self.network_config
            )
            for _ in range(self.n_agents)
        ]

        for _ in range(genome['lifespan']):
            for agent in agents:
                if not agent.alive:
                    continue
                inputs = self._sense(agent, self.canvas)
                agent.step(
                    inputs,
                    genome['weights'],
                    genome['step_scale'],
                    genome['turn_scale']
                )
                self.canvas.record_position(agent.x, agent.y)

                if (agent.x < 0 or agent.x > self.canvas.width or
                        agent.y < 0 or agent.y > self.canvas.height):
                    agent.alive = False

        self._write_svg(agents, output_path)

    def estimate_complexity(self, genome):
        return self.n_agents * genome['lifespan'] * 0.001

    def _write_svg(self, agents, output_path):
        import svgwrite

        dwg = svgwrite.Drawing(
            output_path,
            size=(f'{self.canvas.width}px', f'{self.canvas.height}px'),
            viewBox=f'0 0 {self.canvas.width} {self.canvas.height}'
        )

        # White background
        dwg.add(dwg.rect(
            insert=(0, 0),
            size=(self.canvas.width, self.canvas.height),
            fill='white'
        ))

        for agent in agents:
            if len(agent.path) < 2:
                continue
            
            simplified = simplify_coords(agent.path, 1.0)

            # Draw the agent's path as a polyline
            dwg.add(dwg.polyline(
                points=simplified,
                stroke='black',
                stroke_width=0.5,
                fill='none',
                stroke_linecap='round',
                stroke_linejoin='round'
            ))

        dwg.save()

    def preview(self, output_path, show=True):
        import webbrowser
        import os
        if show:
            webbrowser.open(f'file://{os.path.abspath(output_path)}')

if __name__ == "__main__":
    config = {'n_agents': 50,
              }
    
    seed = NeuralAgentSeed(config)
    genome = seed.random_genome()
    seed.render(genome, 'output.svg')
    seed.preview('output.svg')