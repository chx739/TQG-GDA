"""Explicit dependencies on the frozen R1/R2/R3/R4 experiment implementations."""
from experiments.mutagenicity.r1_图表示与边界支持_20261111.src import data
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src import common as io
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src import evaluate as output
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.score import assert_nested_equal
from experiments.mutagenicity.r3_源风险与置信度约束_20261112.src import protocol as p3
from experiments.mutagenicity.r3_源风险与置信度约束_20261112.src import trajectory as t3
from experiments.mutagenicity.r3_源风险与置信度约束_20261112.src import diagnostics as d3
from experiments.mutagenicity.r3_源风险与置信度约束_20261112.src import constraints
from experiments.mutagenicity.r3_源风险与置信度约束_20261112.src.source_roles import split_source_roles
from experiments.mutagenicity.r4_全方向迁移对照_20261112.src import protocol as p4
from experiments.mutagenicity.r4_全方向迁移对照_20261112.src import training
from experiments.mutagenicity.r4_全方向迁移对照_20261112.src import trajectory as t4
from experiments.mutagenicity.r4_全方向迁移对照_20261112.src import evaluate as e4
from experiments.mutagenicity.r4_全方向迁移对照_20261112.src.run import run_stage
from experiments.common.io import digest, provenance
