import click
import pkg_resources

from clapper.click import AliasedGroup
from click_plugins import with_plugins


@with_plugins(pkg_resources.iter_entry_points("bob.bio.pipeline.cli"))
@click.group(cls=AliasedGroup)
def pipeline():
    """Available pipelines"""
    pass
