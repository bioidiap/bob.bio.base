"""Prints a detailed list of all resources that are registered, including the modules, where they have been registered."""

import os
import warnings

import click

import bob.bio.base


@click.command()
@click.option(
    "--types",
    "-t",
    type=click.Choice(
        ["database", "annotator", "pipeline", "config", "dask"],
        case_sensitive=False,
    ),
    multiple=True,
    default=["database", "annotator", "pipeline", "config", "dask"],
    help="Select the resource types that should be listed.",
)
@click.option(
    "--details",
    "-d",
    is_flag=True,
    default=False,
    help="Prints the complete configuration for all resources",
)
@click.option(
    "--packages",
    "-p",
    multiple=True,
    help="If given, shows only resources defined in the given package(s)",
)
@click.option(
    "--no-strip-dummy",
    "-s",
    is_flag=True,
    default=False,
    help="If given, the dummy elements (usually used for testing purposes only) are **not** removed from the list.",
)
def resources(types, details, packages, no_strip_dummy):
    """Lists the currently registered configurations for this environment."""

    # TODO This needs to be updated!

    kwargs = {"verbose": details, "packages": packages}
    if no_strip_dummy:
        kwargs["strip"] = []

    if "database" in types:
        print(
            "\nList of registered databases (can be used after the --database "
            "option):"
        )
        print(bob.bio.base.list_resources("database", **kwargs))

    if "annotator" in types:
        print(
            "\nList of registered annotators (can be used after the "
            "--annotator option):"
        )
        print(bob.bio.base.list_resources("annotator", **kwargs))

    if "pipeline" in types:
        print(
            "\nList of registered pipelines (can be used after the --pipeline "
            "option):"
        )
        print(bob.bio.base.list_resources("pipeline", **kwargs))

    if "config" in types:
        print(
            "\nList of registered configs. Configs may contain multiple "
            "resources and they also allow chain loading (see bob.extension "
            "docs on chain loading). Configs are used as arguments to commands "
            "such as simple):"
        )
        print(bob.bio.base.list_resources("config", **kwargs))

    if "dask" in types:
        print("\nList of registered dask clients")
        print(
            bob.bio.base.list_resources(
                "client", package_prefix="dask.", **kwargs
            )
        )

    print()


def databases(command_line_parameters=None):
    warnings.warn(
        "The databases command is deprecated. You should not be using it!"
    )
    import argparse

    database_replacement = "%s/.bob_bio_databases.txt" % os.environ["HOME"]

    parser = argparse.ArgumentParser(
        description="Prints a list of directories for registered databases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-D",
        "--database-directories-file",
        metavar="FILE",
        default=database_replacement,
        help="The file, where database directories are stored (to avoid changing the database configurations)",
    )

    args = parser.parse_args(command_line_parameters)

    # get registered databases
    databases = bob.bio.base.utils.resources.database_directories(
        replacements=args.database_directories_file
    )

    # print directories for all databases
    for d in sorted(databases):
        print("\n%s:" % d)

        print("Original data: %s" % databases[d][0])
        if len(databases[d]) > 1:
            print("Annotations: %s" % databases[d][1])
