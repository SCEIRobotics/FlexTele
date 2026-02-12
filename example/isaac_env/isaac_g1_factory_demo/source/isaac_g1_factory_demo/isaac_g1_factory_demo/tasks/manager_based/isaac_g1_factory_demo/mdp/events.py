from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING

from isaacsim.core.utils.extensions import enable_extension

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg

def randomize_visual_texture_prim(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    prim_path: str,
    textures: list[str],
    default_texture: str = "",
    texture_rotation: tuple[float, float] = (0.0, 0.0),
):
    """Randomize the visual texture of bodies on an asset using Replicator API.

    This function randomizes the visual texture of the bodies of the asset using the Replicator API.
    The function samples random textures from the given texture paths and applies them to the bodies
    of the asset. The textures are projected onto the bodies and rotated by the given angles.

    .. note::
        The function assumes that the asset follows the prim naming convention as:
        "{asset_prim_path}/{body_name}/visuals" where the body name is the name of the body to
        which the texture is applied. This is the default prim ordering when importing assets
        from the asset converters in Isaac Lab.

    .. note::
        When randomizing the texture of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    # enable replicator extension if not already enabled
    enable_extension("omni.replicator.core")
    # we import the module here since we may not always need the replicator
    import omni.replicator.core as rep

    # check to make sure replicate_physics is set to False, else raise error
    # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
    #   and the event manager doesn't check in that case.
    if env.cfg.scene.replicate_physics:
        raise RuntimeError(
            "Unable to randomize visual texture material with scene replication enabled."
            " For stable USD-level randomization, please disable scene replication"
            " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
        )

    # convert from radians to degrees
    texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)
    prims_group = rep.get.prims(prim_path)

    with prims_group:
        rep.randomizer.texture(
            textures=textures, project_uvw=True, texture_rotate=rep.distribution.uniform(*texture_rotation)
        )