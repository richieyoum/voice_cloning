
import numpy as np

__mapping_array = None

with open('speaker/speakers.txt') as speakers:
    lines = []
    for line in speakers.readlines():
        if line[0] == ';':
            continue
        lines.append(line)

    rows = [line.split('|') for line in lines]

    __mapping_list = [(int(row[0].strip()), row[1].strip()) for row in rows]

    max_id = max([speaker_id for (speaker_id, _) in __mapping_list])\

    __mapping_array = np.zeros(max_id + 1,)
    for speaker_id, gender in __mapping_list:
        if gender == 'F':
            __mapping_array[speaker_id] = 1
        else:
            __mapping_array[speaker_id] = 2


def get_mapping_array():
    return np.copy(__mapping_array)
