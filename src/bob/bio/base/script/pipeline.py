import click
import pkg_resources

from click_plugins import with_plugins
from exposed.click import AliasedGroup


@with_plugins(pkg_resources.iter_entry_points("bob.bio.pipeline.cli"))
@click.group(cls=AliasedGroup)
def pipeline():
    """Available pipelines"""
    pass
