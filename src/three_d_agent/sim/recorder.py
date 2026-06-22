
from pathlib import Path

import imageio
import numpy as np


class VideoRecorder:
    def __init__(self, path: Path, fps: int = 30) -> None:
        self.path = Path(path)
        self.fps = fps
        self._writer = None
        self._count = 0

    def add_frame(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(self.path, fps=self.fps, codec="libx264")
        self._writer.append_data(frame)
        self._count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
