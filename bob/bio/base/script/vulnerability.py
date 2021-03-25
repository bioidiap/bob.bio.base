"""The main entry for bob.vuln
"""
import click
import pkg_resources
from click_plugins import with_plugins
from bob.extension.scripts.click_helper import AliasedGroup


@with_plugins(pkg_resources.iter_entry_points('bob.vuln.cli'))
@click.group(cls=AliasedGroup)
def vulnerability():
  """Vulnerability analysis related commands."""
  pass
