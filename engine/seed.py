from abc import ABC, abstractmethod

class MendelSeed(ABC):

    @abstractmethod
    def random_genome(self) -> dict:
        """Generate a random starting genome for this environment."""
        pass

    @abstractmethod
    def mutate(self, genome: dict, rate: float) -> dict:
        """Return a mutated copy of the genome."""
        pass

    @abstractmethod
    def crossover(self, genome_a: dict, genome_b: dict) -> dict:
        """Combine two genomes to produce an offspring."""
        pass

    @abstractmethod
    def render(self, genome: dict, output_path: str) -> None:
        """Render the genome to an SVG file at output_path."""
        pass

    @abstractmethod
    def estimate_complexity(self, genome: dict) -> float:
        """Return estimated plot time in seconds."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human readable environment name."""
        pass

    # @property
    # @abstractmethod
    # def genome_schema(self) -> dict:
    #     """Describe the genome structure for display and debugging."""
    #     pass
    
    @property
    def canvas_width(self) -> float:
        return 800.0
    
    @property
    def canvas_height(self) -> float:
        return 800.0