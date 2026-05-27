# engine/agents/neural_agent.py

import math

class NeuralAgent:

    def __init__(self, x, y, heading, network_config):
        self.x = x
        self.y = y
        self.heading = heading
        self.alive = True
        self.age = 0
        self.path = [(x, y)]

        self.input_size = network_config['input_size']
        self.hidden_size = network_config['hidden_size']
        self.output_size = network_config['output_size']

    @property
    def n_weights(self):
        return (
            self.input_size * self.hidden_size +
            self.hidden_size * self.output_size
        )

    def forward(self, inputs, weights):
        """Run inputs through the network, return outputs."""
        # Slice weights into two matrices
        n_w1 = self.input_size * self.hidden_size
        w1 = weights[:n_w1]
        w2 = weights[n_w1:]

        # Hidden layer
        hidden = []
        for i in range(self.hidden_size):
            total = sum(
                inputs[j] * w1[i * self.input_size + j]
                for j in range(self.input_size)
            )
            hidden.append(math.tanh(total))

        # Output layer
        outputs = []
        for i in range(self.output_size):
            total = sum(
                hidden[j] * w2[i * self.hidden_size + j]
                for j in range(self.hidden_size)
            )
            outputs.append(math.tanh(total))

        return outputs

    def step(self, inputs, weights, step_scale, turn_scale):
        """Move the agent one tick given network inputs."""
        turn_rate, speed = self.forward(inputs, weights)
        self.heading += turn_rate * turn_scale
        self.x += math.cos(self.heading) * speed * step_scale
        self.y += math.sin(self.heading) * speed * step_scale
        self.age += 1
        self.path.append((self.x, self.y))