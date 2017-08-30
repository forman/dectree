import time

import numpy as np
import pandas as pd
#from intertidal_flat_classif import Input, Output, apply_rules
#from intertidal_flat_classif_opt import Input, Output, apply_rules
from intertidal_flat_classif_fuz import Input, Output, apply_rules

INPUT_TXT = "verification_input.txt"
EXPECTED_TXT = "verification_expected.txt"
#OUTPUT_TXT = "verification_output.txt"
#OUTPUT_TXT = "verification_output_opt.txt"
OUTPUT_TXT = "verification_output_fuz.txt"

input_frame = pd.read_csv(INPUT_TXT, delimiter='\t', skip_blank_lines=True, comment='#')
output_frame = pd.read_csv(EXPECTED_TXT, delimiter='\t', skip_blank_lines=True, comment='#')

input_frame.sort_values(by="Label")
output_frame.sort_values(by="Label")

# print(input_frame['Label'])
# print(output_frame['Label'])

input_names = [
    ("b1", "sand-tr_abundance"),
    ("b2", "sand-wc_abundance"),
    ("b3", "schatten_abundance"),
    ("b4", "summary_error"),
    ("b5", "steigung_red_nIR"),
    ("b6", "steigung_nIR_SWIR1"),
    ("b7", "flh"),
    ("b8", "ndvi"),
    ("b12", "reflec_483"),
    ("b13", "reflec_561"),
    ("b14", "reflec_655"),
    ("b15", "reflec_865"),
    ("b19", "muschelindex"),
    ("b16", "reflec_1609"),
    ("bsum", "reflec_sum"),
]

output_names = [
    "nodata",
    "Wasser",
    "Schill",
    "Muschel",
    "dense2",
    "dense1",
    "Strand",
    "Sand",
    "Misch",
    "Misch2",
    "Schlick",
    "schlick_t",
    "Wasser2",
]


def to_array(frame, name):
    return np.array(frame[name].values)


expected_class = to_array(output_frame, "Band_1")

dectree_input = Input(expected_class.size)
dectree_output = Output(expected_class.size)

for input_name, column_name in input_names:
    setattr(dectree_input, input_name, to_array(input_frame, column_name))

t0 = time.clock()
apply_rules(dectree_input, dectree_output)
ms_first_time = (time.clock() - t0) * 1000

print('Inputs:')
for input_name, _ in input_names:
    print('{}: {}'.format(input_name, getattr(dectree_input, input_name)))

print('Outputs:')
print('{}: {}'.format('expected_class', expected_class))
for output_name in output_names:
    print('{}: {}'.format(output_name, getattr(dectree_output, output_name)))

frame = pd.DataFrame.from_items(zip(['expected_class'] + [output_name for output_name in output_names],
                                    [expected_class] + [getattr(dectree_output, output_name) for output_name in
                                                        output_names]))

frame.to_csv(path_or_buf=OUTPUT_TXT, sep='\t')


# Performance check:

n = 25
tsum = 0
for i in range(n):
    for input_name, column_name in input_names:
        setattr(dectree_input, input_name, to_array(input_frame, column_name))
    t0 = time.clock()
    apply_rules(dectree_input, dectree_output)
    tsum += time.clock() - t0

ms_per_pixel = (tsum / n / expected_class.size) * 1000
pixel_per_sec = 1000 / ms_per_pixel
print('apply_rules() took {} ms for the first time'.format(ms_first_time))
print('apply_rules() took {} ms per pixel according to {} pixels per second'.format(ms_per_pixel, pixel_per_sec))
