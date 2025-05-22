import open3d as o3d
from PIL import Image

def enableHouModule():
    '''Set up the environment so that "import hou" works.'''
    import sys, os
    sys.path.append("/home/hmittal/hfs19.5.773/houdini/python3.9libs")
    sys.path.append("/home/hmittal/hfs19.5.773/houdini/python3.9libs")
    sys.path.append("/home/hmittal/hfs19.5.773/python/lib")
    if hasattr(sys, "setdlopenflags"):
        old_dlopen_flags = sys.getdlopenflags()
        sys.setdlopenflags(old_dlopen_flags | os.RTLD_GLOBAL)
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory("{}/bin".format(os.environ["HFS"]))
    try:
        import hou
    except ImportError:
        sys.path.append(os.environ['HHP'])
        import hou
    finally:
        if hasattr(sys, "setdlopenflags"):
            sys.setdlopenflags(old_dlopen_flags)

def visualize_simulation(trajectory):
    import hou
    enableHouModule()

    hou.hipFile.clear()
    camera_node = hou.node("/obj").createNode("cam", "teaser_cam_zoom")

    camera_node.parm("resx").set(1280)  # Set resolution width
    camera_node.parm("resy").set(720)  # Set resolution height
    camera_node.parm("focal").set(1)  # Set focal length
    camera_node.parm("aperture").set(1)  # Set aperture size

    # Set camera position and orientation
    camera_node.parmTuple("t").set((-0.0299566, 0.609477, 1.97237))  # Set position (x, y, z)
    camera_node.parmTuple("r").set((-12.1102, 0.00672398, -0.01411))  # Set rotation (rx, ry, rz)

    # Set camera display
    camera_node.setDisplayFlag(True)

    # set background
    bg_path = "/home/hmittal/diff-sim/houdini_projects/Teaser/room.jpg"
    camera_node.parm("vm_background").set(bg_path)

    # Add lights
    light_node = hou.node("/obj").createNode("hlight")
    light_node.parm("light_type").set("distant")
    light_node.parmTuple("t").set((6.24513, 1.71573, 9.09945))
    light_node.parmTuple("r").set((-4.76341, 38.0068, 8.67156e-05))
    light_node.parm("light_intensity").set(3.45)
    light_node.parm("light_exposure").set(0)

    # Add box node
    geo_node2 = hou.node("/obj").createNode("geo")
    box_node = geo_node2.createNode("box")
    geo_node2.parm("sx").set(1)
    geo_node2.parm("sy").set(0.4)
    geo_node2.parm("sz").set(0.5)
    box_node.parmTuple("size").set((4, 0.5, 4))
    box_node.setDisplayFlag(True)
    box_node.setRenderFlag(True)
    box_node.parmTuple("t").set((0, -0.25, 0))
    geo_node2.cook(force=True)

    quick_material_node = geo_node2.createNode("labs::quickmaterial", "labs_quick_material")
    quick_material_node.parm("usemikkt").set(True)
    quick_material_node.parm("mMaterialEntries").set(1)
    quick_material_node.parm("principledshader_basecolor_texture_1").set("/home/hmittal/diff-sim/houdini_projects/Teaser/pexels-fwstudio-33348-129733.jpg")
    quick_material_node.parm("principledshader_rough_1").set(0.181)
    quick_material_node.parm("principledshader_ior_1").set(0.55)
    quick_material_node.parm("principledshader_occlusion_textureIntensity_1").set(1)
    quick_material_node.setInput(0, box_node, 0)
    quick_material_node.setDisplayFlag(True)
    quick_material_node.setRenderFlag(True)

    # Render
    render_node = hou.node("/out").createNode("ifd")
    render_node.parm("camera").set(camera_node.path())

    data = trajectory
    num_frames = 0
    for idx in range(0, data.shape[0], 200):
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(data[idx])
        geo_node = None

        try:
            geo_node = hou.node('/obj').createNode('geo')

            geo = geo_node.createNode("file")
            geo.parm("file").set(f"points.ply")   # Load PLY points file here
            geo_node.cook(force=True)

            # particlefluidsurface_node.parms()
            particlefluidsurface_node = geo_node.createNode("particlefluidsurface")
            particlefluidsurface_node.parm("particlesep").set(0.01)
            particlefluidsurface_node.parm("voxelsize").set(0.2)
            particlefluidsurface_node.parm("influenceradius").set(3)
            particlefluidsurface_node.parm("surfacedistance").set(1)
            particlefluidsurface_node.parm("resamplingiterations").set(5)
            particlefluidsurface_node.parm("surferosion").set(0.8)
            particlefluidsurface_node.parm("dilateoffset").set(5.76)
            particlefluidsurface_node.parm("smoothoperation").set("Mean Curvature Flow")
            particlefluidsurface_node.parm("erodeoffset").set(5.76)
            particlefluidsurface_node.setInput(0, geo)
            geo_node.cook(force=True)

            # render the mesh
            particlefluidsurface_node.setDisplayFlag(True)
            particlefluidsurface_node.setRenderFlag(True)
            geo_node.cook(force=True)

            color_node = geo_node.createNode("color")
            color_node.setInput(0, particlefluidsurface_node)
            color_node.parmTuple("color").set(tuple([0.25, 0, 0.5]))
            color_node.setDisplayFlag(True)
            color_node.setRenderFlag(True)
            geo_node.cook(force=True)

            render_node.parm("vm_picture").set(f"img_{idx}.jpg")
            render_node.render()
            num_frames += 1

        except ImportError:
            print ("ImportError: Could not import hou module. Please check your Houdini installation.")

    # Create GIF from saved images
    frames_l = []
    for i in range(num_frames):
        frames_l.append(f"img_{i}.jpg")
    frame_duration = 0.08
    frames = [Image.open(image).convert("RGBA") for image in frames_l]
    frames[0].save('visualization.gif', save_all=True, append_images=frames[1:], duration=frame_duration * 1000,loop=0)
    print("GIF saved at: ", 'visualization.gif')