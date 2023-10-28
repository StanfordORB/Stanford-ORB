from typing import Dict, List, Any
import abc


class BasePipeline(abc.ABC):
    def __init__(self):
        pass

    def test_inverse_rendering(self, scene: str, overwrite: bool = False) -> Dict[str, Any]:
        raise NotImplementedError()

    def test_new_view(self, scene: str, overwrite: bool = False) -> List[Dict[str, str]]:
        raise NotImplementedError()

    def test_new_light(self, scene: str, overwrite: bool = False) -> List[Dict[str, str]]:
        raise NotImplementedError()

    def test_geometry(self, scene: str, overwrite: bool = False) -> List[Dict[str, str]]:
        raise NotImplementedError()

    def test_material(self, scene: str, overwrite: bool = False) -> List[Dict[str, str]]:
        raise NotImplementedError()

    def test_shape(self, scene: str, overwrite: bool = False) -> Dict[str, str]:
        raise NotImplementedError()
