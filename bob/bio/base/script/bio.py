"""The main entry for bob.bio (click-based) scripts.
"""
import click
import pkg_resources
from click_plugins import with_plugins


@with_plugins(pkg_resources.iter_entry_points('bob.bio.cli'))
@click.group()
def bio():
    """Entry for bob.bio commands."""
    pass
