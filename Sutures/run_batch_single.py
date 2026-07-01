from abaqus import *
from abaqusConstants import *
from caeModules import *
from odbAccess import openOdb
import os, sys, csv, json, time, traceback

WORK_DIR    = r'C:\temp\Scraba'
JSON_DIR    = os.path.join(WORK_DIR, 'geometries')
RESULTS_CSV = os.path.join(WORK_DIR, 'results.csv')
LOG_FILE    = os.path.join(WORK_DIR, 'batch_log.txt')

sys.path.insert(0, WORK_DIR)
import abaqus_script as ASCRIPT

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = '[%s] %s' % (ts, msg)
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def get_completed():
    done = set()
    if not os.path.exists(RESULTS_CSV):
        return done
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            if row.get('status') == 'completed':
                done.add(row['job_name'])
    return done

def process_one(json_path):
    curve_engine, meta = ASCRIPT.read_geometry(json_path)
    curve_abq = ASCRIPT.convert_coords(curve_engine)
    stem = os.path.splitext(os.path.basename(json_path))[0]
    job_name = 'Job-' + stem
    ASCRIPT.build_model(curve_abq, job_name, meta)
    results = ASCRIPT.extract_results(job_name, meta)
    if results is None:
        raise RuntimeError('extraction returned None')
    ASCRIPT.write_results(RESULTS_CSV, job_name, meta, results)
    return results

def main():
    if not os.path.exists(JSON_DIR):
        print('ERROR: geometries folder not found: %s' % JSON_DIR)
        return

    json_files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith('.json'))
    if not json_files:
        print('No JSON files in %s' % JSON_DIR)
        return

    completed = get_completed()
    pending = [f for f in json_files
               if ('Job-' + os.path.splitext(f)[0]) not in completed]

    log('Batch run started (single session)')
    log('Total JSON files : %d' % len(json_files))
    log('Already completed: %d' % len(completed))
    log('Pending          : %d' % len(pending))
    log('-' * 50)

    ok = 0; fail = 0
    for i, jf in enumerate(pending):
        jpath = os.path.join(JSON_DIR, jf)
        job_name = 'Job-' + os.path.splitext(jf)[0]
        t0 = time.time()
        try:
            with open(jpath) as f:
                meta = json.load(f)
            log('[%d/%d] %s  type=%s N=%s k=%s'
                % (i+1, len(pending), jf, meta.get('geo_type','?'),
                   meta.get('N','?'), meta.get('k','?')))
            res = process_one(jpath)
            log('   OK  Fmax=%.3g  k=%.3g  U=%.3g  smax=%.3g  (%.1fs)'
                % (res['Fmax'], res['k_stiff'], res['U_energy'],
                   res['sigma_max'], time.time()-t0))
            ok += 1
        except Exception as e:
            fail += 1
            log('   FAILED: %s' % str(e))
            with open(LOG_FILE, 'a') as f:
                f.write(traceback.format_exc() + '\n')
            # write a failed-status row so resume skips it
            write_header = not os.path.exists(RESULTS_CSV)
            with open(RESULTS_CSV, 'a') as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(['job_name','geo_type','N','k','alpha',
                                'R_min','arc_length','Fmax','k_stiff',
                                'U_energy','sigma_max','n_frames','status'])
                w.writerow([job_name, meta.get('geo_type',''),
                            meta.get('N',''), meta.get('k',''),
                            meta.get('alpha',''), meta.get('R_min',''),
                            meta.get('arc_length',''), '', '', '', '', '',
                            'failed'])

    log('-' * 50)
    log('Batch complete.  OK=%d  FAILED=%d' % (ok, fail))
    log('Results: %s' % RESULTS_CSV)

main()