#!/usr/bin/env python3
"""
Run the whole RAVDESS evaluation pipeline on Windows, macOS or Linux:

  1. check that the data, the EfficientFace weights and the checkpoint are in place
  2. create the .venv virtual environment and install requirements.txt
  3. preprocessing: extract_faces.py -> extract_audios.py -> create_annotations.py
  4. python main.py --no_train --no_val --result_path <CHECKPOINT_FOLDER>

Expected layout (RAVDESS sits next to the repository folder):

  <parent>/
  |-- RAVDESS/
  |   |-- EfficientFace_Trained_on_AffectNet7.pth.tar
  |   |-- Actor_01/  01-01-...-01.mp4, 02-01-...-01.mp4, 03-01-...-01.wav, ...
  |   |-- ...
  |   `-- Actor_24/
  `-- multimodal-emotion-recognition/   (this repository)
      `-- run_pipeline.py

Usage (run from anywhere with Python 3.9-3.12):

  macOS / Linux:  python3 run_pipeline.py --checkpoint_dir PATH_TO_CHECKPOINT_FOLDER
  Windows:        py -3.12 run_pipeline.py --checkpoint_dir PATH_TO_CHECKPOINT_FOLDER

Add --check_only to run only the checks.
"""
import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
PREPROC = os.path.join(REPO, 'ravdess_preprocessing')
VENV = os.path.join(REPO, '.venv')
VENV_PY = os.path.join(VENV, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(VENV, 'bin', 'python')
PRETRAIN_NAME = 'EfficientFace_Trained_on_AffectNet7.pth.tar'
# main.py loads <result_path>/<dataset>_<model>_<sample_duration>_best<fold>.pth
CHECKPOINT_NAME = 'RAVDESS_multimodalcnn_15_best0.pth'
# Actor split used by create_annotations.py (0-based actor index -> actor number - 1)
TEST_ACTORS, VAL_ACTORS = {1, 2, 3, 4}, {5, 6, 7, 8}
SUPPORTED_PY = ((3, 9), (3, 12))  # torch 2.2 / facenet-pytorch 2.6 wheels exist for these


# ----------------------------------------------------------------------------- helpers
def banner(text):
    print('\n' + '=' * 70 + '\n' + text + '\n' + '=' * 70, flush=True)


def ok(msg):
    print('  [OK]   ' + msg)


def warn(msg):
    print('  [WARN] ' + msg)


def run(cmd, cwd=REPO, env=None):
    print('  $ ' + ' '.join('"%s"' % c if ' ' in c else c for c in cmd), flush=True)
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        sys.exit('\nERROR: command failed (exit code %d): %s' % (result.returncode, ' '.join(cmd)))


def py_version_ok(version):
    return SUPPORTED_PY[0] <= tuple(version[:2]) <= SUPPORTED_PY[1]


def actor_number(folder_name):
    m = re.search(r'\d+', folder_name)
    return int(m.group()) if m else None


# ----------------------------------------------------------------------------- checks
def check_setup(args):
    """Return (errors, pretrain_path). Prints a line for every check."""
    errors = []
    banner('STEP 1/4  Checking files and folders')

    # Python
    v = sys.version_info
    if os.path.exists(VENV_PY):
        ok('virtual environment found: %s' % VENV)
    elif py_version_ok(v):
        ok('Python %d.%d.%d can be used to create the virtual environment' % v[:3])
    else:
        errors.append('Python %d.%d is not supported (torch 2.2 needs 3.9-3.12). Re-run with a supported '
                      'Python, e.g. "py -3.12 run_pipeline.py ..." on Windows or "python3.12 run_pipeline.py ..." on macOS.' % v[:2])

    # Repository files
    for rel in ['requirements.txt', 'main.py', 'ravdess_preprocessing/extract_faces.py',
                'ravdess_preprocessing/extract_audios.py', 'ravdess_preprocessing/create_annotations.py']:
        if os.path.isfile(os.path.join(REPO, rel)):
            ok('found %s' % rel)
        else:
            errors.append('missing repository file: %s' % rel)

    # RAVDESS folder
    ravdess = args.ravdess_dir
    if not os.path.isdir(ravdess):
        errors.append('RAVDESS folder not found: %s (it should sit next to the repository folder, '
                      'or pass --ravdess_dir)' % ravdess)
        return errors, None
    ok('RAVDESS folder: %s' % ravdess)

    # EfficientFace weights: inside RAVDESS, falling back to the repository root
    pretrain = args.pretrain_path
    if pretrain is None:
        candidates = [os.path.join(ravdess, PRETRAIN_NAME), os.path.join(REPO, PRETRAIN_NAME)]
        pretrain = next((c for c in candidates if os.path.isfile(c)), None)
        if pretrain is None:
            errors.append('%s not found in %s (download it from https://github.com/zengqunzhao/EfficientFace, '
                          '"Pre-trained Models", Nov. 12 2021 AffectNet-7 link)' % (PRETRAIN_NAME, ravdess))
        elif os.path.dirname(pretrain) != ravdess:
            warn('%s not in RAVDESS; using the copy at %s' % (PRETRAIN_NAME, pretrain))
        else:
            ok('EfficientFace weights: %s' % pretrain)
    elif os.path.isfile(pretrain):
        ok('EfficientFace weights: %s' % pretrain)
    else:
        errors.append('--pretrain_path file not found: %s' % pretrain)

    # Actor folders
    actors = {}
    for name in sorted(os.listdir(ravdess)):
        path = os.path.join(ravdess, name)
        num = actor_number(name)
        if os.path.isdir(path) and num is not None and 1 <= num <= 24:
            if num in actors:
                errors.append('two folders for actor %d: %s and %s' % (num, os.path.basename(actors[num]), name))
            actors[num] = path

    print('\n  Actor folders (videos = 01-/02-*.mp4, audio = 03-*.wav):')
    for num in range(1, 25):
        role = 'test' if num in TEST_ACTORS else 'val' if num in VAL_ACTORS else 'train'
        needed = num in TEST_ACTORS  # --no_train --no_val only uses the test actors
        if num not in actors:
            msg = 'actor %02d (%s): folder missing' % (num, role)
            errors.append(msg) if needed else warn(msg + ' (only needed for training/validation)')
            continue
        files = os.listdir(actors[num])
        videos = [f for f in files if f.endswith('.mp4') and f[:3] in ('01-', '02-')]
        wavs = {f for f in files if f.endswith('.wav') and f.startswith('03-') and 'croppad' not in f}
        no_audio = [v for v in videos if '03' + v[2:-4] + '.wav' not in wavs]
        line = 'actor %02d (%-5s) %-10s %3d videos, %3d audio' % (num, role, os.path.basename(actors[num]), len(videos), len(wavs))
        problems = []
        if not videos:
            problems.append('no .mp4 videos')
        if not wavs:
            problems.append('no 03-*.wav audio')
        if no_audio:
            problems.append('%d videos without matching audio, e.g. %s needs %s'
                            % (len(no_audio), no_audio[0], '03' + no_audio[0][2:-4] + '.wav'))
        if not problems:
            ok(line)
        elif needed:
            errors.append(line + ': ' + '; '.join(problems))
        else:
            warn(line + ': ' + '; '.join(problems) + ' (only needed for training/validation)')

    # Checkpoint folder
    print()
    ckpt_dir = args.checkpoint_dir
    if not os.path.isdir(ckpt_dir):
        errors.append('checkpoint folder not found: %s' % ckpt_dir)
    elif os.path.isfile(os.path.join(ckpt_dir, CHECKPOINT_NAME)):
        ok('checkpoint: %s' % os.path.join(ckpt_dir, CHECKPOINT_NAME))
    else:
        found = [os.path.relpath(os.path.join(d, f), ckpt_dir)
                 for d, _, fs in os.walk(ckpt_dir) for f in fs if f.endswith(('.pth', '.pth.tar'))]
        hint = ('; .pth files found there: %s. main.py only loads a file named %s, so point --checkpoint_dir at '
                'the folder holding it or rename it' % (', '.join(found[:5]), CHECKPOINT_NAME)) if found else ''
        errors.append('%s not found in %s%s' % (CHECKPOINT_NAME, ckpt_dir, hint))

    return errors, pretrain


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint_dir', required=True,
                        help='folder containing %s (passed to main.py as --result_path)' % CHECKPOINT_NAME)
    parser.add_argument('--ravdess_dir', default=os.path.join(os.path.dirname(REPO), 'RAVDESS'),
                        help='RAVDESS data folder (default: RAVDESS next to the repository)')
    parser.add_argument('--pretrain_path', default=None,
                        help='EfficientFace weights (default: RAVDESS/%s)' % PRETRAIN_NAME)
    parser.add_argument('--fusion', default='ia', choices=['ia', 'it', 'lt'],
                        help='fusion type the checkpoint was trained with (default: ia)')
    parser.add_argument('--n_threads', type=int, default=min(16, os.cpu_count() or 1),
                        help='data loader workers for main.py')
    parser.add_argument('--check_only', action='store_true', help='only run the checks')
    parser.add_argument('--skip_install', action='store_true', help='do not run pip install')
    parser.add_argument('--skip_preprocessing', action='store_true', help='reuse existing preprocessed files')
    args = parser.parse_args()
    args.ravdess_dir = os.path.abspath(args.ravdess_dir)
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if args.pretrain_path:
        args.pretrain_path = os.path.abspath(args.pretrain_path)

    errors, pretrain = check_setup(args)
    if errors:
        print('\nSetup is not complete. Fix the following and run again:')
        for e in errors:
            print('  [FAIL] ' + e)
        sys.exit(1)
    print('\nAll checks passed.')
    if args.check_only:
        return

    # Step 2: virtual environment + requirements
    banner('STEP 2/4  Virtual environment and requirements.txt')
    if not os.path.exists(VENV_PY):
        run([sys.executable, '-m', 'venv', VENV])
    venv_version = subprocess.run([VENV_PY, '-c', 'import sys; print(sys.version_info[0], sys.version_info[1])'],
                                  capture_output=True, text=True).stdout.split()
    if not venv_version or not py_version_ok(tuple(map(int, venv_version))):
        sys.exit('ERROR: %s uses Python %s, which torch 2.2 does not support. Delete the .venv folder and re-run '
                 'this script with Python 3.9-3.12.' % (VENV, '.'.join(venv_version) or '?'))
    ok('using %s (Python %s)' % (VENV_PY, '.'.join(venv_version)))
    if args.skip_install:
        ok('skipping pip install (--skip_install)')
    else:
        run([VENV_PY, '-m', 'pip', 'install', '--upgrade', 'pip'])
        run([VENV_PY, '-m', 'pip', 'install', '-r', os.path.join(REPO, 'requirements.txt')])
    run([VENV_PY, '-c', 'import torch, torchvision, librosa, cv2, facenet_pytorch, numpy; '
                        'print("  imports OK: torch", torch.__version__, "numpy", numpy.__version__)'])

    # Step 3: preprocessing
    banner('STEP 3/4  Preprocessing (faces, audio, annotations)')
    annotation_path = os.path.join(PREPROC, 'annotations.txt')
    if args.skip_preprocessing:
        ok('skipping preprocessing (--skip_preprocessing)')
    else:
        env = dict(os.environ, RAVDESS_ROOT=args.ravdess_dir)
        for script in ['extract_faces.py', 'extract_audios.py', 'create_annotations.py']:
            print('\n  -> %s' % script)
            run([VENV_PY, script], cwd=PREPROC, env=env)

    if not os.path.isfile(annotation_path):
        sys.exit('ERROR: %s was not created.' % annotation_path)
    counts, missing = defaultdict(int), []
    with open(annotation_path) as f:
        for line in f:
            video, audio, _, subset = line.rstrip('\n').split(';')
            counts[subset] += 1
            missing += [p for p in (video, audio) if subset == 'testing' and not os.path.isfile(p)]
    ok('annotations.txt: %d training, %d validation, %d testing'
       % (counts['training'], counts['validation'], counts['testing']))
    if counts['testing'] == 0:
        sys.exit('ERROR: annotations.txt has no testing samples; check that actors %s have videos and audio.'
                 % sorted(TEST_ACTORS))
    if missing:
        sys.exit('ERROR: %d preprocessed test files are missing, e.g. %s' % (len(missing), missing[0]))

    # Step 4: evaluation
    banner('STEP 4/4  python main.py --no_train --no_val')
    run([VENV_PY, 'main.py', '--no_train', '--no_val',
         '--result_path', args.checkpoint_dir,
         '--annotation_path', annotation_path,
         '--pretrain_path', pretrain,
         '--fusion', args.fusion,
         '--n_threads', str(args.n_threads)])

    result_file = os.path.join(args.checkpoint_dir, 'test_set_bestval.txt')
    banner('DONE')
    if os.path.isfile(result_file):
        with open(result_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        print('  Test accuracy (latest line of %s):\n  %s' % (result_file, lines[-1] if lines else '(empty)'))
    print('\n  To use the environment yourself:')
    print('    Windows:      .venv\\Scripts\\activate' if os.name == 'nt' else '    macOS/Linux:  source .venv/bin/activate')


if __name__ == '__main__':
    main()
