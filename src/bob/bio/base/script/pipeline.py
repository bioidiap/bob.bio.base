import click
import pkg_resources

from click_plugins import with_plugins

from bob.extension.scripts.click_helper import AliasedGroup


@with_plugins(pkg_resources.iter_entry_points("bob.bio.pipeline.cli"))
@click.group(cls=AliasedGroup)
def pipeline():
    """Available pipelines"""
    pass
