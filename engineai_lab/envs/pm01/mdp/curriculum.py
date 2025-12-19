from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from .scene import terrain_levels_curriculum


@configclass
class CurriculumCfg:
    terrain_levels = CurriculumTermCfg(func=terrain_levels_curriculum)
