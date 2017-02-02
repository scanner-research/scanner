import xlrd
import sys
import json

CUTTING_DIR = '/h/wcrichto/multcorrs'

CLASSES = [
    'master',
    'SRS',
    'SRS',
    'OTS',
    'ph',
    'SRS',
    'moving',
    'char',
    'moving',
    'veh',
    'POV',
    'reaction',
    'action',
    'insert',
    'cutaway',
    'env',
    'est',
    'montage',
    'solo',
    'other']

def workbook(name):
    return xlrd.open_workbook('{}/{}mcorr.xls'.format(CUTTING_DIR, name))

def main():
    movie = sys.argv[1] if len(sys.argv) > 1 else 'inception'
    book = workbook(movie)
    assert book.nsheets == 4

    shot_scale_sheet = book.sheet_by_index(0)
    shot_class_sheet = book.sheet_by_index(1)
    nrows = shot_scale_sheet.nrows
    frames = []
    for row in range(nrows):
        frame = shot_scale_sheet.cell(row, 0).value
        scale = shot_scale_sheet.cell(row, 14).value
        if not isinstance(frame, float) or not isinstance(scale, float): continue
        clas = -1
        for col in range(4, 18):
            flag = shot_class_sheet.cell(row, col).value
            if isinstance(flag, float):
                clas = col-4
        frame = int(round(frame))
        scale = int(round(scale))
        frames.append((frame, scale, clas))
    with open('{}.txt'.format(movie), 'w') as f:
        f.write(json.dumps(frames))

if __name__ == "__main__":
    main()
