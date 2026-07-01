from abaqus import *
from abaqusConstants import *
from caeModules import *
from odbAccess import openOdb
import json
import os
import sys
import csv
import math
import numpy as np

# Configuration — edit these to match your setup

WORK_DIR     = r'C:\temp\Scraba'
RESULTS_CSV  = os.path.join(WORK_DIR, 'results.csv')
JSON_DIR     = os.path.join(WORK_DIR, 'geometries')  # where JSON files live

# Specimen geometry (matches your manual model)
SPEC_WIDTH   = 10.0    # half-width of each specimen half (mm)
SPEC_HEIGHT  = 10.0    # total height (mm)  y: -5 to +5
CX           = 0.0     # interface centerline x

# Material (from your inp file)
E_MODULUS    = 2.0e11  # Pa
POISSON      = 0.3

# Contact
FRICTION_COEF = 0.0    # mu — Malik et al. 2017 value
                        # your manual model used 0.0 (frictionless)
                        # change to 0.0 if you want to match exactly

# Loading
DISPLACEMENT  = 0.5    # mm tensile pullout on right edge

# Step settings
INITIAL_INC   = 0.1
MAX_INC_TIME  = 1.0
MIN_INC       = 1e-9
MAX_INC_SIZE  = 0.1
TOTAL_INC     = 1000

# Mesh seed (global 0.5, interface refined for jigsaw neck curvature)
MESH_GLOBAL   = 0.5
MESH_INTERFACE= 0.08   # fine seed near suture interface to resolve tight necks


# Helper: read JSON geometry file

def read_geometry(json_path):
    """
    Read curve_points from JSON exported by Cell 9 of the Colab notebook.
    Returns list of (x, y) tuples in physical coordinates (mm).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    curve_points = [(pt[0], pt[1]) for pt in data['curve_points']]
    meta = {
        'geo_type'  : data.get('geo_type',   'unknown'),
        'N'         : data.get('N',          0),
        'k'         : data.get('k',          0),
        'alpha'     : data.get('alpha',      0),
        'R_min'     : data.get('R_min',      0),
        'arc_length': data.get('arc_length', 0),
    }
    return curve_points, meta


# Helper: convert geometry engine coords → Abaqus coords

def convert_coords(curve_points, spec_height=SPEC_HEIGHT):
    """
    Geometry engine uses:  x in [0, 10],  y in [0, 10]
    Abaqus model uses:     x centered at 0, y in [-5, +5]

    Transform:
        x_abq = x_engine - 5.0   (shift so centerline is at x=0)
        y_abq = y_engine - 5.0   (shift so y=0 is at specimen mid-height)

    Also snaps the first/last points exactly onto the bottom/top edges
    (y = -H2 and y = +H2) so the closed profile is watertight in Abaqus.
    """
    H2 = spec_height / 2.0
    converted = []
    for (x, y) in curve_points:
        x_abq = x - spec_height / 2.0
        y_abq = y - spec_height / 2.0
        converted.append((x_abq, y_abq))

    # snap endpoints exactly to bottom and top edges
    x0, y0 = converted[0]
    converted[0]  = (x0, -H2)     # bottom edge
    xN, yN = converted[-1]
    converted[-1] = (xN,  H2)     # top edge
    return converted


# Main model builder

def part_edge_array(part, edge_list):
    """
    Convert a Python list of edge objects into the Abaqus EdgeArray type
    that Surface() and seedEdgeBySize() require.
    Uses findAt on each edge's pointOn coordinate.
    """
    pts = tuple((e.pointOn[0],) for e in edge_list)
    return part.edges.findAt(*pts)


def _face_x_extent(part, face):
    """
    Return (xmin, xmax) of a face by scanning the vertices of its edges.
    Used to determine which outer edge (x=-W or x=+W) a face reaches.
    """
    xs = []
    for e in face.getEdges():
        edge = part.edges[e]
        for v in edge.getVertices():
            xs.append(part.vertices[v].pointOn[0][0])
    if not xs:
        # fallback to face centroid
        return face.pointOn[0][0], face.pointOn[0][0]
    return min(xs), max(xs)


def build_model(curve_points_abq, job_name, meta):
    """
    Build complete Abaqus model from spline curve points.
    Creates two-part specimen, assigns contact, BCs, step, output requests.
    """

    # ── create new model ─────────────────────────────────────────────────────
    model_name = 'Suture-Model'
    if model_name in mdb.models:
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)

    # ── specimen dimensions ───────────────────────────────────────────────────
    W  = SPEC_WIDTH    # half-width: left half x in [-W, 0], right half x in [0, W]
    H2 = SPEC_HEIGHT / 2.0   # half-height: y in [-H2, +H2]

    # ── subsample curve points for Abaqus sketch ─────────────────────────────
    # Abaqus s.Spline() works best with 50-150 points
    n_pts = len(curve_points_abq)
    step  = max(1, n_pts // 120)
    pts_sketch = list(curve_points_abq[::step])
    if list(pts_sketch[0]) != list(curve_points_abq[0]):
        pts_sketch = [curve_points_abq[0]] + pts_sketch
    if list(pts_sketch[-1]) != list(curve_points_abq[-1]):
        pts_sketch = pts_sketch + [curve_points_abq[-1]]
    pts_sketch = [tuple(p) for p in pts_sketch]

    # spline runs bottom (y=-H2) to top (y=+H2) along the interface
    p_bottom = pts_sketch[0]     # near (0, -H2)
    p_top    = pts_sketch[-1]    # near (0, +H2)

    left_part  = model.Part(name='Left',  dimensionality=TWO_D_PLANAR,
                            type=DEFORMABLE_BODY)
    right_part = model.Part(name='Right', dimensionality=TWO_D_PLANAR,
                            type=DEFORMABLE_BODY)

    # Build each half by: make full rectangle, partition with spline,
    # then REMOVE the unwanted face. The face to KEEP is the one whose
    # bounding box reaches the correct outer edge (x=-W for left half,
    # x=+W for right half). This is robust regardless of where the
    # partitioned-face centroids land relative to the centreline.

    def build_half(part, keep_outer_x):
        """
        keep_outer_x = -W to keep the half touching the LEFT outer edge,
                       +W to keep the half touching the RIGHT outer edge.
        """
        sk = model.ConstrainedSketch(name=part.name+'Rect', sheetSize=50.0)
        sk.rectangle(point1=(-W, -H2), point2=(W, H2))
        part.BaseShell(sketch=sk)

        # partition the single face with the spline
        csk = model.ConstrainedSketch(name=part.name+'Cut', sheetSize=50.0)
        csk.Spline(points=pts_sketch)
        part.PartitionFaceBySketch(faces=part.faces, sketch=csk)

        # now identify the two faces by which outer edge they reach
        keep_face   = None
        remove_face = None
        for f in part.faces:
            xmin, xmax = _face_x_extent(part, f)
            touches_left  = abs(xmin - (-W)) < 0.01
            touches_right = abs(xmax -  ( W)) < 0.01
            if keep_outer_x < 0:   # want left half
                if touches_left and not touches_right:
                    keep_face = f
                else:
                    remove_face = f
            else:                  # want right half
                if touches_right and not touches_left:
                    keep_face = f
                else:
                    remove_face = f

        if remove_face is not None:
            part.RemoveFaces(faceList=(remove_face,), deleteCells=False)

    build_half(left_part,  -W)
    build_half(right_part,  W)

    # ── verify geometry: each half should be ~half the full rectangle area ───
    full_area = (2.0 * W) * (2.0 * H2)   # total rectangle area
    try:
        a_left  = left_part.getArea(left_part.faces)
        a_right = right_part.getArea(right_part.faces)
        print("  GEOMETRY CHECK:")
        print("    full rectangle area = %.2f mm^2" % full_area)
        print("    left  part area     = %.2f mm^2" % a_left)
        print("    right part area     = %.2f mm^2" % a_right)
        print("    left+right          = %.2f (should ~= full)" % (a_left+a_right))
        if abs((a_left + a_right) - full_area) > 0.05 * full_area:
            print("    *** WARNING: parts overlap or have gap! ***")
        else:
            print("    OK: parts are complementary halves")
    except Exception as e:
        print("  (area check skipped: %s)" % e)

    # ── material ──────────────────────────────────────────────────────────────
    mat = model.Material(name='Material-1')
    mat.Elastic(table=((E_MODULUS, POISSON),))

    # ── sections ──────────────────────────────────────────────────────────────
    model.HomogeneousSolidSection(name='Section-1',
                                   material='Material-1',
                                   thickness=None)

    # create a face set on each part, then assign the section to it
    left_part.Set(name='All-Left',  faces=left_part.faces)
    right_part.Set(name='All-Right', faces=right_part.faces)

    left_part.SectionAssignment(
        region=left_part.sets['All-Left'],
        sectionName='Section-1')
    right_part.SectionAssignment(
        region=right_part.sets['All-Right'],
        sectionName='Section-1')

    # ── assembly ─────────────────────────────────────────────────────────────
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)

    left_inst  = assembly.Instance(name='Left-1',
                                    part=left_part,
                                    dependent=ON)
    right_inst = assembly.Instance(name='Right-1',
                                    part=right_part,
                                    dependent=ON)

    # ── interface surfaces (created from PART edges, before meshing) ─────────
    # The interface is the spline edge. The left part's interface edge is the
    # one NOT on the outer rectangle boundary (not at x=-W, y=+-H2).
    # We identify it as the edge whose midpoint x is closest to the centerline
    # region (the spline swings around x=0 but its average is near 0).

    def find_interface_edges(part, outer_x):
        """
        Return the interface edge(s) — those not lying on the 3 straight
        rectangle sides (x=outer_x, y=+H2, y=-H2).
        """
        iface = []
        for e in part.edges:
            pt = e.pointOn[0]
            x, y = pt[0], pt[1]
            on_outer_x = abs(x - outer_x) < 0.05
            on_top     = abs(y - H2) < 0.05
            on_bottom  = abs(y + H2) < 0.05
            if not (on_outer_x or on_top or on_bottom):
                iface.append(e)
        return iface

    left_iface  = find_interface_edges(left_part,  -W)
    right_iface = find_interface_edges(right_part,  W)

    left_part.Surface(name='IfaceL',
                      side1Edges=part_edge_array(left_part, left_iface))
    right_part.Surface(name='IfaceR',
                       side1Edges=part_edge_array(right_part, right_iface))

    # ── surface interaction ───────────────────────────────────────────────────
    # frictionless tangential + hard normal — matches working manual model
    model.ContactProperty('IntProp-1')
    model.interactionProperties['IntProp-1'].TangentialBehavior(
        formulation=FRICTIONLESS)
    model.interactionProperties['IntProp-1'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON,
        constraintEnforcementMethod=DEFAULT)

    model.SurfaceToSurfaceContactStd(
        name='Int-1',
        createStepName='Initial',
        main=left_inst.surfaces['IfaceL'],
        secondary=right_inst.surfaces['IfaceR'],
        sliding=FINITE,
        thickness=ON,
        interactionProperty='IntProp-1',
        adjustMethod=NONE,
        initialClearance=OMIT,
        datumAxis=None,
        clearanceRegion=None)

    # ── mesh ─────────────────────────────────────────────────────────────────
    # FREE quad mesh with advancing-front algorithm avoids the distorted
    # triangular elements that cause contact chatter at the jigsaw necks.
    for part, iface in ((left_part, left_iface), (right_part, right_iface)):
        part.seedPart(size=MESH_GLOBAL, deviationFactor=0.1, minSizeFactor=0.1)
        if iface:
            part.seedEdgeBySize(
                edges=part_edge_array(part, iface),
                size=MESH_INTERFACE, deviationFactor=0.1,
                minSizeFactor=0.1, constraint=FINER)
        # CPS8R quadratic elements (matches working manual model) — mid-side
        # nodes represent the curved jigsaw bulbs smoothly so contact slides
        # cleanly instead of catching on linear-element facets.
        part.setMeshControls(
            regions=part.faces, elemShape=QUAD,
            algorithm=ADVANCING_FRONT)
        elem_type = mesh.ElemType(elemCode=CPS8R, elemLibrary=STANDARD)
        part.setElementType(regions=(part.faces,), elemTypes=(elem_type,))
        part.generateMesh()

    # ── node sets for BCs (created on PARTS, then referenced via instance) ───
    # For dependent instances the mesh lives on the part, so the BC sets
    # must be created on the part using edge-based node selection.

    # Left part: outer edge is the straight vertical line at x = -W
    left_outer_edge = left_part.edges.getByBoundingBox(
        xMin=-W-0.05, xMax=-W+0.05,
        yMin=-H2-0.05, yMax=H2+0.05)
    left_part.Set(name='LeftEdgeSet', edges=left_outer_edge)

    # Right part: outer edge is the straight vertical line at x = +W
    right_outer_edge = right_part.edges.getByBoundingBox(
        xMin=W-0.05, xMax=W+0.05,
        yMin=-H2-0.05, yMax=H2+0.05)
    right_part.Set(name='RightEdgeSet', edges=right_outer_edge)

    # reference these part sets through the assembly instances for BCs
    left_bc_region  = left_inst.sets['LeftEdgeSet']
    right_bc_region = right_inst.sets['RightEdgeSet']

    # ── step ─────────────────────────────────────────────────────────────────
    # step settings match the working manual model (no stabilization)
    model.StaticStep(
        name='Step-1',
        previous='Initial',
        nlgeom=ON,
        maxNumInc=TOTAL_INC,
        initialInc=INITIAL_INC,
        minInc=MIN_INC,
        maxInc=MAX_INC_SIZE)

    # ── boundary conditions ───────────────────────────────────────────────────
    # BC-1: fix left edge (U1=0, U2=0)
    model.DisplacementBC(
        name='BC-Fixed',
        createStepName='Initial',
        region=left_bc_region,
        u1=SET, u2=SET, ur3=SET,
        amplitude=UNSET, distributionType=UNIFORM,
        fieldName='', localCsys=None)

    # BC-2: pull right edge (U1=+0.5mm)
    model.DisplacementBC(
        name='BC-Pull',
        createStepName='Step-1',
        region=right_bc_region,
        u1=DISPLACEMENT, u2=SET, ur3=SET,
        amplitude=UNSET, distributionType=UNIFORM,
        fieldName='', localCsys=None)

    # ── output requests ───────────────────────────────────────────────────────
    # Field output — stress, displacement, reaction force
    model.fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'U', 'RF', 'E'))

    # History output — RF1 on left edge (for F-U curve)
    model.HistoryOutputRequest(
        name='H-RF',
        createStepName='Step-1',
        variables=('RF1',),
        region=left_bc_region,
        sectionPoints=DEFAULT,
        rebar=EXCLUDE,
        frequency=1)

    # History output — U1 on right edge (displacement)
    model.HistoryOutputRequest(
        name='H-U',
        createStepName='Step-1',
        variables=('U1',),
        region=right_bc_region,
        sectionPoints=DEFAULT,
        rebar=EXCLUDE,
        frequency=1)

    # ── create and submit job ─────────────────────────────────────────────────
    job = mdb.Job(
        name=job_name,
        model=model_name,
        description=f"Suture {meta['geo_type']} N={meta['N']} k={meta['k']}",
        type=ANALYSIS,
        atTime=None,
        waitMinutes=0,
        waitHours=0,
        queue=None,
        memory=90,
        memoryUnits=PERCENTAGE,
        getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE,
        echoPrint=OFF,
        modelPrint=OFF,
        contactPrint=OFF,
        historyPrint=OFF,
        userSubroutine='',
        scratch='',
        resultsFormat=ODB,
        multiprocessingMode=DEFAULT,
        numCpus=1,
        numGPUs=0)

    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()

    return job_name


# Post-processor: extract Fmax, k, U, sigma_max from .odb

def extract_results(job_name, meta):
    """
    Open .odb and extract mechanical response metrics.
    Returns dict with Fmax, k_stiff, U_energy, sigma_max.
    """
    odb_path = os.path.join(WORK_DIR, job_name + '.odb')
    odb      = openOdb(path=odb_path, readOnly=True)

    # find the step robustly — don't assume exact key 'Step-1'
    step_keys = list(odb.steps.keys())
    print("  available steps in odb: %s" % step_keys)
    if not step_keys:
        print("  ERROR: no steps in odb")
        odb.close()
        return None
    step = odb.steps[step_keys[-1]]   # last (analysis) step

    # if the analysis diverged, the step exists but has 0 frames —
    # return None cleanly so the batch marks it failed and continues
    n_frames = len(step.frames)
    if n_frames == 0:
        print("  ERROR: step has 0 frames (analysis did not converge)")
        odb.close()
        return None

    # diagnostic — how far did the analysis actually get?
    print("  ODB diagnostics:")
    print("    number of frames in step: %d" % n_frames)
    print("    history regions: %d" % len(step.historyRegions.keys()))
    for rk in step.historyRegions.keys():
        outs = step.historyRegions[rk].historyOutputs.keys()
        print("      region %s -> outputs: %s" % (rk, list(outs)))

    # ── extract F-U curve from history output ────────────────────────────────
    # RF1 from left edge — summed over all nodes
    rf_data = {}   # time → total RF1
    u_data  = {}   # time → U1

    for region_key in step.historyRegions.keys():
        region = step.historyRegions[region_key]

        # reaction force (left edge)
        if 'RF1' in region.historyOutputs:
            d = region.historyOutputs['RF1'].data
            if d:
                for (time, val) in d:
                    rf_data[time] = rf_data.get(time, 0.0) + val

        # displacement (right edge)
        if 'U1' in region.historyOutputs:
            d = region.historyOutputs['U1'].data
            if d:
                for (time, val) in d:
                    u_data[time] = val   # any node on right edge

    # sort by time
    times = sorted(rf_data.keys())
    RF    = [abs(rf_data[t]) for t in times]   # absolute value (tensile)
    U1    = [u_data.get(t, t * DISPLACEMENT) for t in times]

    # ── fallback: if history output empty, extract RF from field output ───────
    # sum RF1 over the left-edge node set across all frames
    if not RF:
        print("  History output empty — falling back to field-output RF")
        rf_frames = []
        u_frames  = []
        try:
            left_set = odb.rootAssembly.instances['LEFT-1'].nodeSets['LEFTEDGESET']
        except Exception:
            left_set = None
        try:
            right_set = odb.rootAssembly.instances['RIGHT-1'].nodeSets['RIGHTEDGESET']
        except Exception:
            right_set = None

        for frame in step.frames:
            t = frame.frameValue
            # reaction force sum on left edge
            if left_set is not None and 'RF' in frame.fieldOutputs:
                rf_sub = frame.fieldOutputs['RF'].getSubset(region=left_set)
                total = 0.0
                for v in rf_sub.values:
                    total += v.data[0]   # RF1 component
                rf_frames.append(abs(total))
            # displacement on right edge
            if right_set is not None and 'U' in frame.fieldOutputs:
                u_sub = frame.fieldOutputs['U'].getSubset(region=right_set)
                if len(u_sub.values) > 0:
                    u_frames.append(u_sub.values[0].data[0])
                else:
                    u_frames.append(t * DISPLACEMENT)
            else:
                u_frames.append(t * DISPLACEMENT)

        RF = rf_frames
        U1 = u_frames

    # ── Fmax ─────────────────────────────────────────────────────────────────
    Fmax = max(RF) if RF else 0.0

    # ── stiffness k = slope of initial linear portion ────────────────────────
    # use first 20% of displacement range
    n_linear = max(2, len(RF) // 5)
    if len(RF) >= 2 and U1[-1] > 1e-10:
        dF = RF[n_linear] - RF[0]
        dU = U1[n_linear] - U1[0]
        k_stiff = dF / dU if abs(dU) > 1e-12 else 0.0
    else:
        k_stiff = 0.0

    # ── strain energy U = area under F-U curve (trapezoidal) ─────────────────
    U_energy = 0.0
    for i in range(1, len(RF)):
        dU = U1[i] - U1[i-1]
        U_energy += 0.5 * (RF[i] + RF[i-1]) * dU

    # ── sigma_max = max von Mises stress (from field output) ─────────────────
    sigma_max = 0.0
    last_frame = step.frames[-1]
    if 'MISES' in last_frame.fieldOutputs:
        mises = last_frame.fieldOutputs['MISES']
        vals  = [v.data for v in mises.values]
        sigma_max = max(vals) if vals else 0.0
    elif 'S' in last_frame.fieldOutputs:
        # fall back to computing Mises from stress tensor
        S_field = last_frame.fieldOutputs['S']
        for v in S_field.values:
            s = v.data   # [S11, S22, S33, S12]
            mises_val = math.sqrt(
                0.5 * ((s[0]-s[1])**2 + (s[1]-s[2])**2 +
                        (s[2]-s[0])**2 + 6*s[3]**2))
            if mises_val > sigma_max:
                sigma_max = mises_val

    n_frames = len(step.frames)
    odb.close()

    return {
        'Fmax'     : round(Fmax,      4),
        'k_stiff'  : round(k_stiff,   4),
        'U_energy' : round(U_energy,  6),
        'sigma_max': round(sigma_max, 4),
        'n_frames' : n_frames,
    }


# CSV writer

def write_results(results_csv, job_name, meta, results):
    """Append one row to results CSV."""
    row = {
        'job_name'  : job_name,
        'geo_type'  : meta['geo_type'],
        'N'         : meta['N'],
        'k'         : meta['k'],
        'alpha'     : meta['alpha'],
        'R_min'     : meta['R_min'],
        'arc_length': meta['arc_length'],
        'Fmax'      : results['Fmax'],
        'k_stiff'   : results['k_stiff'],
        'U_energy'  : results['U_energy'],
        'sigma_max' : results['sigma_max'],
        'n_frames'  : results['n_frames'],
        'status'    : 'completed',
    }
    write_header = not os.path.exists(results_csv)
    with open(results_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"  Written to {results_csv}")


# MAIN — called by Abaqus

def main():
    """
    Main entry point. Reads JSON path from command line argument.
    Usage: abaqus cae noGUI=abaqus_script.py -- path/to/geometry.json

    Note: Abaqus handles the '--' separator inconsistently across versions.
    Some versions strip it, some keep it. So instead of relying on '--',
    we scan ALL command-line arguments for the first one ending in '.json'.
    """
    # robust JSON path detection — scan all args for a .json file
    json_path = None
    for arg in sys.argv:
        if arg.lower().endswith('.json'):
            json_path = arg
            break

    # if still not found, check if a relative path was passed after --
    if json_path is None and '--' in sys.argv:
        idx = sys.argv.index('--')
        if idx + 1 < len(sys.argv):
            json_path = sys.argv[idx + 1]

    # diagnostic: print all args so we can see what Abaqus passed
    if not json_path or not os.path.exists(json_path):
        print("ERROR: No valid JSON file path found.")
        print("Arguments received by script:")
        for i, a in enumerate(sys.argv):
            print("   argv[%d] = %s" % (i, a))
        print("Usage: abaqus cae noGUI=abaqus_script.py -- geometry.json")
        # try resolving relative to WORK_DIR/geometries as fallback
        if json_path:
            fallback = os.path.join(JSON_DIR, os.path.basename(json_path))
            if os.path.exists(fallback):
                print("Found fallback path: %s" % fallback)
                json_path = fallback
            else:
                return
        else:
            return

    # derive job name from JSON filename
    json_basename = os.path.splitext(os.path.basename(json_path))[0]
    job_name      = f"Job-{json_basename}"

    print(f"\n{'='*60}")
    print(f"  Processing: {json_basename}")
    print(f"{'='*60}")

    # read geometry
    curve_points_engine, meta = read_geometry(json_path)
    print(f"  Type: {meta['geo_type']}  N={meta['N']}  "
          f"k={meta['k']}  α={meta['alpha']}")
    print(f"  Curve points: {len(curve_points_engine)}")

    # convert to Abaqus coordinates
    curve_points_abq = convert_coords(curve_points_engine)
    print(f"  x range: [{min(p[0] for p in curve_points_abq):.3f}, "
          f"{max(p[0] for p in curve_points_abq):.3f}] mm")
    print(f"  y range: [{min(p[1] for p in curve_points_abq):.3f}, "
          f"{max(p[1] for p in curve_points_abq):.3f}] mm")

    # build and run
    print(f"\n  Building model ...")
    build_model(curve_points_abq, job_name, meta)
    print(f"  Simulation complete.")

    # extract results
    print(f"  Extracting results ...")
    results = extract_results(job_name, meta)
    if results is None:
        print("  ERROR: could not extract results from odb")
        return
    print(f"  Fmax     = {results['Fmax']:.4f} N")
    print(f"  k_stiff  = {results['k_stiff']:.4f} N/mm")
    print(f"  U_energy = {results['U_energy']:.6f} N·mm")
    print(f"  sigma_max= {results['sigma_max']:.4f} Pa")

    # write to CSV
    write_results(RESULTS_CSV, job_name, meta, results)
    print(f"\n  Done. ✓")


# run
main()