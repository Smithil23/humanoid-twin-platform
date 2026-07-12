"""Pipeline stages produce a loadable, contact-ready model."""
import mujoco
import pytest

from htp import PlatformConfig, UrdfPipeline

CFG = "configs/star1.yaml"


@pytest.fixture(scope="module")
def model():
    mjcf = UrdfPipeline(PlatformConfig.load(CFG)).build()
    return mujoco.MjModel.from_xml_string(mjcf)


def test_floating_base(model):
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE


def test_floor_and_feet_geoms(model):
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        for g in range(model.ngeom)
    ]
    assert "floor" in names
    assert model.ngeom >= 3  # floor + 2 sole boxes


def test_every_hinge_has_actuator(model):
    hinges = sum(
        model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        for j in range(model.njnt)
    )
    assert model.nu == hinges


def test_mass_is_plausible(model):
    total = float(model.body_mass.sum())
    assert 30 < total < 200, f"unexpected robot mass {total} kg"


def test_visual_mode_keeps_visual_meshes():
    import xml.etree.ElementTree as ET

    from htp.pipeline import UrdfPipeline

    cfg = PlatformConfig.load("configs/star1_visual.yaml")
    pipe = UrdfPipeline(cfg)
    tree = ET.parse(cfg.robot.urdf)
    pipe.sanitize(tree)
    root = tree.getroot()
    visual_meshes = sum(
        1 for link in root.iter("link")
        for v in link.findall("visual") if v.find(".//mesh") is not None
    )
    collision_meshes = sum(
        1 for link in root.iter("link")
        for c in link.findall("collision") if c.find(".//mesh") is not None
    )
    assert visual_meshes > 50          # meshes kept for rendering
    assert collision_meshes == 0       # contact still from sole boxes
    assert root.find("mujoco/compiler") is not None
