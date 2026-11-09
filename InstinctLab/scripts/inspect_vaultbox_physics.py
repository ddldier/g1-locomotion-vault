import argparse

# ---------------------------------------------------------
# 1. Start Isaac Sim before importing USD / Isaac Lab modules.
# ---------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="Instinct-Unified-Transition-PPO-Play-G1-v0",
)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------
# 2. Import Isaac Lab and create the actual environment.
# ---------------------------------------------------------
env = None

try:
    import gymnasium as gym
    import omni.usd

    from isaaclab.utils.dict import print_dict
    from isaaclab_tasks.utils import parse_env_cfg

    import instinctlab.tasks  # Register tasks

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=False,
    )

    print("\n========== BOX CONFIG ==========")
    print("Box size:", env_cfg.scene.vault_box.spawn.size)
    print("Initial position:", env_cfg.scene.vault_box.init_state.pos)
    print("Rigid props:")
    print_dict(env_cfg.scene.vault_box.spawn.rigid_props)
    print("Collision props:")
    print_dict(env_cfg.scene.vault_box.spawn.collision_props)
    print("Physics material:")
    print_dict(env_cfg.scene.vault_box.spawn.physics_material)

    env = gym.make(args.task, cfg=env_cfg)
    raw_env = env.unwrapped

    # -----------------------------------------------------
    # 3. Get the actual USD Stage.
    # -----------------------------------------------------
    stage = omni.usd.get_context().get_stage()

    print("\n========== LOCATE VAULT BOX ==========")

    boxes = []

    for prim in stage.Traverse():
        if prim.GetName() == "VaultBox":
            boxes.append(prim)

    if not boxes:
        raise RuntimeError("No VaultBox prim found in USD Stage!")

    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema



    for box in boxes[:2]:
        print("\n" + "=" * 80)
        print("PRIM PATH:", box.GetPath())
        print("PRIM TYPE:", box.GetTypeName())
        print("APPLIED SCHEMAS:", box.GetAppliedSchemas())

        # Inspect every descendant of VaultBox.
        print("\n========== ALL VAULTBOX DESCENDANTS ==========")

        for child in Usd.PrimRange(box):
            print("\n" + "-" * 70)
            print("PATH:", child.GetPath())
            print("TYPE:", child.GetTypeName())
            print("SCHEMAS:", child.GetAppliedSchemas())

            if child.IsA(UsdGeom.Cube):
                cube = UsdGeom.Cube(child)
                print("CUBE SIZE:", cube.GetSizeAttr().Get())

            if child.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(child)
                points = mesh.GetPointsAttr().Get()
                print("MESH POINT COUNT:", len(points) if points else 0)
                print("MESH EXTENT:", mesh.GetExtentAttr().Get())

            for attr in child.GetAttributes():
                name = attr.GetName()
                if any(k in name.lower() for k in
                       ["physics:", "physx", "collision", "material"]):
                    print(
                        "ATTR:", name,
                        "=", repr(attr.Get()),
                        "AUTHORED:", attr.HasAuthoredValueOpinion()
                    )

            for rel in child.GetRelationships():
                print(
                    "REL:", rel.GetName(),
                    "TARGETS:", rel.GetTargets()
                )

	
        # -------------------------------------------------
        # 4. Read authored and schema attributes.
        # -------------------------------------------------
        print("\n---------- USD / PHYSX ATTRIBUTES ----------")

        for attr in sorted(
            box.GetAttributes(),
            key=lambda x: x.GetName()
        ):
            name = attr.GetName()

            if any(key in name.lower() for key in [
                "physics:",
                "physx",
                "collision",
                "material",
            ]):
                try:
                    value = attr.Get()
                    authored = attr.HasAuthoredValueOpinion()
                    print(
                        f"{name} = {value!r} "
                        f"[authored={authored}]"
                    )
                except Exception as e:
                    print(name, "READ ERROR:", repr(e))

        # -------------------------------------------------
        # 5. Check rigid body and collision APIs.
        # -------------------------------------------------
        print("\n---------- SCHEMA CHECK ----------")

        for api_cls in [
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.MassAPI,
            UsdPhysics.CollisionAPI,
            PhysxSchema.PhysxRigidBodyAPI,
            PhysxSchema.PhysxCollisionAPI,
        ]:
            try:
                api = api_cls(box)
                print(
                    api_cls.__name__,
                    "APPLIED:",
                    api.GetPrim().IsValid(),
                )
            except Exception as e:
                print(api_cls.__name__, repr(e))

        # -------------------------------------------------
        # 6. Read actual geometry.
        # -------------------------------------------------
        print("\n---------- GEOMETRY ----------")

        if box.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(box)
            print("Cube size:", cube.GetSizeAttr().Get())

        xform = UsdGeom.Xformable(box)
        print("Local transform:", xform.GetLocalTransformation())

        # -------------------------------------------------
        # 7. Find bound physics materials.
        # -------------------------------------------------
        print("\n---------- MATERIAL BINDING ----------")

        try:
            binding_api = UsdShade.MaterialBindingAPI(box)
            material, relationship = binding_api.ComputeBoundMaterial()

            if material:
                material_prim = material.GetPrim()
                print("Material path:", material_prim.GetPath())
                print("Material schemas:", material_prim.GetAppliedSchemas())

                for attr in material_prim.GetAttributes():
                    name = attr.GetName()
                    if "physics" in name.lower() or "physx" in name.lower():
                        print(name, "=", attr.Get())

                for api_cls in [
                    UsdPhysics.MaterialAPI,
                    PhysxSchema.PhysxMaterialAPI,
                ]:
                    try:
                        api = api_cls(material_prim)
                        print(
                            api_cls.__name__,
                            "APPLIED:",
                            api.GetPrim().IsValid(),
                        )
                    except Exception as e:
                        print(api_cls.__name__, repr(e))
            else:
                print("No bound material found.")

        except Exception as e:
            print("Material inspection error:", repr(e))

    print("\n========== INSPECTION COMPLETE ==========")

finally:
    if env is not None:
        env.close()

    simulation_app.close()
