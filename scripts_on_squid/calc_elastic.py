# 機械特性を計算するためのファイル
import click
from mlptools.analyzer.elastic import ElasticConstantsCalculator

@click.command()
@click.option(
    "--target_dir",
    type=str,
    required=True,
    help="Path to dir where n2p2 calculation was performed"
)
@click.option(
    "--elastic_template_dir",
    type=str,
    required=True,
    help="Path to dir where template for elastic calculation with LAMMPS exists"
)
def main(target_dir, elastic_template_dir):
    calculator = ElasticConstantsCalculator(
        path2n2p2_result=target_dir
        path2template=elastic_template_dir
    )