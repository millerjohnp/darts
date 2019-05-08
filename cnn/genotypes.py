from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)

AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

ASHA200 = Genotype(normal=[('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('none', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('max_pool_3x3', 3), ('none', 1), ('skip_connect', 0), ('sep_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])


# Genotype similarity experiments
GENOTYPE_0 = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 3), ('none', 4), ('avg_pool_3x3', 0)], normal_concat=[2,3,4,5], reduce=[('dil_conv_3x3', 0), ('none', 1), ('skip_connect', 2), ('none', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 4), ('dil_conv_3x3', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_1 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 2), ('none', 4)], normal_concat=[2,3,4,5], reduce=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('none', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1), ('none', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_2 = Genotype(normal=[('sep_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('none', 3), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 3)], normal_concat=[2,3,4,5], reduce=[('none', 0), ('dil_conv_3x3', 1), ('none', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('skip_connect', 2), ('dil_conv_3x3', 3), ('none', 1)], reduce_concat=[2,3,4,5])
GENOTYPE_3 = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('none', 3), ('dil_conv_5x5', 0), ('skip_connect', 0), ('dil_conv_3x3', 3)], normal_concat=[2,3,4,5], reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('none', 0), ('skip_connect', 2), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=[2,3,4,5])
GENOTYPE_4 = Genotype(normal=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 4)], normal_concat=[2,3,4,5], reduce=[('none', 0), ('skip_connect', 1), ('dil_conv_3x3', 2), ('skip_connect', 1), ('none', 2), ('none', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], reduce_concat=[2,3,4,5])
GENOTYPE_5 = Genotype(normal=[('none', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 3), ('dil_conv_3x3', 2), ('skip_connect', 4), ('avg_pool_3x3', 0)], normal_concat=[2,3,4,5], reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('none', 1), ('avg_pool_3x3', 3), ('sep_conv_5x5', 1), ('max_pool_3x3', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_6 = Genotype(normal=[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 3), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('dil_conv_3x3', 0)], normal_concat=[2,3,4,5], reduce=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_7 = Genotype(normal=[('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('skip_connect', 2), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=[2,3,4,5], reduce=[('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('none', 2), ('none', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('skip_connect', 0), ('avg_pool_3x3', 1)], reduce_concat=[2,3,4,5])
GENOTYPE_8 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('none', 0), ('skip_connect', 3), ('max_pool_3x3', 0), ('skip_connect', 4), ('dil_conv_3x3', 3)], normal_concat=[2,3,4,5], reduce=[('none', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)], reduce_concat=[2,3,4,5])
GENOTYPE_9 = Genotype(normal=[('sep_conv_5x5', 1), ('none', 0), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1), ('skip_connect', 2)], normal_concat=[2,3,4,5], reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_10 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('none', 0), ('none', 1)], normal_concat=[2,3,4,5], reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 4), ('sep_conv_5x5', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_11 = Genotype(normal=[('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('none', 0), ('dil_conv_5x5', 4), ('skip_connect', 3)], normal_concat=[2,3,4,5], reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 3), ('none', 1), ('avg_pool_3x3', 1), ('none', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_12 = Genotype(normal=[('none', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 0)], normal_concat=[2,3,4,5], reduce=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('none', 2), ('avg_pool_3x3', 3), ('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_13 = Genotype(normal=[('skip_connect', 1), ('none', 0), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3), ('max_pool_3x3', 1), ('max_pool_3x3', 4), ('none', 3)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 1), ('none', 0), ('skip_connect', 2), ('sep_conv_3x3', 2), ('dil_conv_5x5', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_14 = Genotype(normal=[('none', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 0), ('sep_conv_5x5', 4)], normal_concat=[2,3,4,5], reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_3x3', 3), ('skip_connect', 2), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_15 = Genotype(normal=[('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1), ('none', 3), ('avg_pool_3x3', 4)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('none', 1), ('none', 2), ('avg_pool_3x3', 2), ('none', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_16 = Genotype(normal=[('max_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 3), ('max_pool_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2)], normal_concat=[2,3,4,5], reduce=[('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('dil_conv_5x5', 3)], reduce_concat=[2,3,4,5])
GENOTYPE_17 = Genotype(normal=[('sep_conv_5x5', 1), ('none', 0), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 4)], normal_concat=[2,3,4,5], reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('none', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 3), ('sep_conv_3x3', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_18 = Genotype(normal=[('dil_conv_5x5', 1), ('none', 0), ('dil_conv_3x3', 1), ('none', 2), ('skip_connect', 3), ('none', 2), ('skip_connect', 2), ('none', 1)], normal_concat=[2,3,4,5], reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_5x5', 3), ('avg_pool_3x3', 3), ('dil_conv_5x5', 2)], reduce_concat=[2,3,4,5])
GENOTYPE_19 = Genotype(normal=[('none', 1), ('max_pool_3x3', 0), ('none', 2), ('sep_conv_3x3', 1), ('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 0), ('avg_pool_3x3', 1)], reduce_concat=[2,3,4,5])
GENOTYPE_20 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 4), ('avg_pool_3x3', 1)], normal_concat=[2,3,4,5], reduce=[('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('none', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('skip_connect', 0), ('avg_pool_3x3', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_21 = Genotype(normal=[('max_pool_3x3', 1), ('skip_connect', 0), ('none', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 3), ('none', 3), ('skip_connect', 2)], normal_concat=[2,3,4,5], reduce=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('skip_connect', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 2), ('avg_pool_3x3', 3), ('avg_pool_3x3', 4), ('max_pool_3x3', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_22 = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 1)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_23 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 4), ('sep_conv_5x5', 3)], normal_concat=[2,3,4,5], reduce=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('none', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('sep_conv_3x3', 1)], reduce_concat=[2,3,4,5])
GENOTYPE_24 = Genotype(normal=[('none', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 1), ('none', 4)], normal_concat=[2,3,4,5], reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('none', 2), ('dil_conv_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_25 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('none', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('none', 3), ('max_pool_3x3', 4), ('avg_pool_3x3', 2)], normal_concat=[2,3,4,5], reduce=[('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 4), ('dil_conv_5x5', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_26 = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 3), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3)], reduce_concat=[2,3,4,5])
GENOTYPE_27 = Genotype(normal=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('none', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 4)], normal_concat=[2,3,4,5], reduce=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 3), ('skip_connect', 3), ('max_pool_3x3', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_28 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 4)], normal_concat=[2,3,4,5], reduce=[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 4), ('skip_connect', 2)], reduce_concat=[2,3,4,5])
GENOTYPE_29 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 1), ('skip_connect', 2), ('none', 1), ('sep_conv_5x5', 0), ('none', 0), ('sep_conv_5x5', 1)], normal_concat=[2,3,4,5], reduce=[('sep_conv_5x5', 1), ('none', 0), ('none', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1)], reduce_concat=[2,3,4,5])
GENOTYPE_30 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('none', 0), ('skip_connect', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('none', 0)], normal_concat=[2,3,4,5], reduce=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 3), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_31 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('none', 2), ('skip_connect', 2), ('none', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 3)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 1), ('avg_pool_3x3', 4), ('dil_conv_5x5', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_32 = Genotype(normal=[('none', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 2)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3), ('avg_pool_3x3', 2), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3)], reduce_concat=[2,3,4,5])
GENOTYPE_33 = Genotype(normal=[('avg_pool_3x3', 1), ('none', 0), ('sep_conv_5x5', 2), ('none', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_3x3', 2)], normal_concat=[2,3,4,5], reduce=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], reduce_concat=[2,3,4,5])
GENOTYPE_34 = Genotype(normal=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 2), ('dil_conv_3x3', 0)], normal_concat=[2,3,4,5], reduce=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 1), ('sep_conv_5x5', 3), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_35 = Genotype(normal=[('skip_connect', 1), ('dil_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('none', 1), ('none', 0), ('max_pool_3x3', 4)], normal_concat=[2,3,4,5], reduce=[('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('none', 1), ('max_pool_3x3', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_36 = Genotype(normal=[('avg_pool_3x3', 0), ('none', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('max_pool_3x3', 4), ('max_pool_3x3', 1)], normal_concat=[2,3,4,5], reduce=[('skip_connect', 1), ('none', 0), ('none', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('dil_conv_3x3', 2), ('none', 1), ('dil_conv_5x5', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_37 = Genotype(normal=[('sep_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('none', 1), ('skip_connect', 3), ('avg_pool_3x3', 3), ('max_pool_3x3', 1)], normal_concat=[2,3,4,5], reduce=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('skip_connect', 3), ('dil_conv_5x5', 0), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2)], reduce_concat=[2,3,4,5])
GENOTYPE_38 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 3), ('avg_pool_3x3', 3), ('skip_connect', 4)], normal_concat=[2,3,4,5], reduce=[('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_39 = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('none', 0), ('dil_conv_3x3', 3), ('avg_pool_3x3', 1), ('sep_conv_3x3', 4), ('none', 0)], normal_concat=[2,3,4,5], reduce=[('skip_connect', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('none', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 4), ('max_pool_3x3', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_40 = Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 1), ('none', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4)], normal_concat=[2,3,4,5], reduce=[('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2)], reduce_concat=[2,3,4,5])
GENOTYPE_41 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 3), ('avg_pool_3x3', 2)], normal_concat=[2,3,4,5], reduce=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('none', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_42 = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('none', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_3x3', 2), ('skip_connect', 2), ('skip_connect', 3)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('none', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2), ('none', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_43 = Genotype(normal=[('skip_connect', 0), ('none', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('none', 3), ('sep_conv_5x5', 1)], normal_concat=[2,3,4,5], reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 2), ('none', 3), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_44 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('skip_connect', 0), ('max_pool_3x3', 2), ('dil_conv_3x3', 3), ('skip_connect', 4), ('skip_connect', 3)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 1), ('none', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 2), ('none', 1), ('dil_conv_5x5', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_45 = Genotype(normal=[('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('skip_connect', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0)], normal_concat=[2,3,4,5], reduce=[('sep_conv_3x3', 0), ('none', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=[2,3,4,5])
GENOTYPE_46 = Genotype(normal=[('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1)], normal_concat=[2,3,4,5], reduce=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0)], reduce_concat=[2,3,4,5])
GENOTYPE_47 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3), ('skip_connect', 2), ('dil_conv_3x3', 4)], normal_concat=[2,3,4,5], reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 2), ('skip_connect', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 4)], reduce_concat=[2,3,4,5])
GENOTYPE_48 = Genotype(normal=[('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('none', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3), ('none', 2), ('sep_conv_5x5', 3)], normal_concat=[2,3,4,5], reduce=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('none', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 3), ('dil_conv_5x5', 2)], reduce_concat=[2,3,4,5])
GENOTYPE_49 = Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('none', 3), ('none', 1), ('max_pool_3x3', 4)], normal_concat=[2,3,4,5], reduce=[('none', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2), ('dil_conv_5x5', 4), ('sep_conv_5x5', 1)], reduce_concat=[2,3,4,5])
