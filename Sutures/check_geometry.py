from abaqus import *
from abaqusConstants import *
from caeModules import *
import json, os, sys

WORK_DIR = r'C:\temp\Scraba'
SPEC_WIDTH = 10.0
SPEC_HEIGHT = 10.0

json_path = None
for arg in sys.argv:
    if arg.lower().endswith('.json'):
        json_path = arg
        break

with open(json_path) as f:
    data = json.load(f)
cp = [(p[0], p[1]) for p in data['curve_points']]

H2 = SPEC_HEIGHT / 2.0
W  = SPEC_WIDTH
conv = []
for (x, y) in cp:
    conv.append((x - SPEC_HEIGHT/2.0, y - SPEC_HEIGHT/2.0))
conv[0]  = (conv[0][0],  -H2)
conv[-1] = (conv[-1][0],  H2)

n = len(conv)
stepn = max(1, n // 120)
pts = list(conv[::stepn])
if list(pts[0]) != list(conv[0]):   pts = [conv[0]] + pts
if list(pts[-1]) != list(conv[-1]): pts = pts + [conv[-1]]
pts = [tuple(p) for p in pts]

logf = open(os.path.join(WORK_DIR, 'geo_check.txt'), 'w')
def log(m):
    print(m)
    logf.write(m + '\n')
    logf.flush()

def face_x_extent(part, face):
    xs = []
    for e in face.getEdges():
        edge = part.edges[e]
        for v in edge.getVertices():
            xs.append(part.vertices[v].pointOn[0][0])
    if not xs:
        return face.pointOn[0][0], face.pointOn[0][0]
    return min(xs), max(xs)

log("="*50)
log("GEOMETRY DIAGNOSTIC v2 (outer-edge face selection)")
log("="*50)
log("spline x range: [%.3f, %.3f]" % (min(p[0] for p in pts), max(p[0] for p in pts)))

m = mdb.Model(name='GeoCheck')

def build_half(name, keep_outer_x):
    p = m.Part(name=name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    sk = m.ConstrainedSketch(name=name+'R', sheetSize=50.0)
    sk.rectangle(point1=(-W,-H2), point2=(W,H2))
    p.BaseShell(sketch=sk)
    csk = m.ConstrainedSketch(name=name+'C', sheetSize=50.0)
    csk.Spline(points=pts)
    p.PartitionFaceBySketch(faces=p.faces, sketch=csk)

    log("\n%s: after partition, faces=%d" % (name, len(p.faces)))
    keep_face = None; remove_face = None
    for f in p.faces:
        xmin, xmax = face_x_extent(p, f)
        tl = abs(xmin-(-W))<0.01; tr = abs(xmax-(W))<0.01
        log("   face at centroid x=%.2f  extent=[%.2f,%.2f]  touchesL=%s touchesR=%s area=%.2f"
            % (f.pointOn[0][0], xmin, xmax, tl, tr, p.getArea((f,))))
        if keep_outer_x < 0:
            if tl and not tr: keep_face=f
            else: remove_face=f
        else:
            if tr and not tl: keep_face=f
            else: remove_face=f
    if remove_face is not None:
        p.RemoveFaces(faceList=(remove_face,), deleteCells=False)
        log("%s: KEPT half, final area=%.2f" % (name, p.getArea(p.faces)))
    else:
        log("%s: ERROR could not identify face to remove" % name)
    return p

build_half('Left',  -W)
build_half('Right',  W)

# assemble to visually check overlap
a = m.rootAssembly
a.Instance(name='L', part=m.parts['Left'],  dependent=ON)
a.Instance(name='R', part=m.parts['Right'], dependent=ON)

mdb.saveAs(os.path.join(WORK_DIR, 'geo_check.cae'))
log("\nSaved geo_check.cae")
log("DONE")
logf.close()