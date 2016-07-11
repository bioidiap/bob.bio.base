"""Prints a detailed list of all resources that are registered, including the modules, where they have been registered."""

from __future__ import print_function
import bob.bio.base
import os

def resources(command_line_parameters = None):

  import argparse
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--types", '-t', nargs = '+',
                      choices = ('d', 'database', 'p', 'preprocessor', 'e', 'extractor', 'a', 'algorithm', 'g', 'grid'),
                      default = ('d', 'p', 'e', 'a', 'g'),
                      help = "Select the resource types that should be listed.")

  parser.add_argument("--details", '-d', action='store_true', help = "Prints the complete configuration for all resources")

  parser.add_argument("--packages", '-p', nargs='+', help = "If given, shows only resources defined in the given package(s)")

  parser.add_argument("--no-strip-dummy", '-s', action = 'store_true',
                      help = "If given, the dummy elements (usually used for testing purposes only) are **not** removed from the list.")

  args = parser.parse_args(command_line_parameters)

  kwargs = {'verbose' : args.details, "packages" : args.packages}
  if args.no_strip_dummy:
    kwargs['strip'] = []


  if 'd' in args.types or 'database' in args.types:
    print ("\nList of registered databases:")
    print (bob.bio.base.list_resources('database', **kwargs))

  if 'p' in args.types or 'preprocessor' in args.types:
    print ("\nList of registered preprocessors:")
    print (bob.bio.base.list_resources('preprocessor', **kwargs))

  if 'e' in args.types or 'extractor' in args.types:
    print ("\nList of registered extractors:")
    print (bob.bio.base.list_resources('extractor', **kwargs))

  if 'a' in args.types or 'algorithm' in args.types:
    print ("\nList of registered algorithms:")
    print (bob.bio.base.list_resources('algorithm', **kwargs))

  if 'g' in args.types or 'grid' in args.types:
    print ("\nList of registered grid configurations:")
    print (bob.bio.base.list_resources('grid', **kwargs))

  print()

def databases(command_line_parameters = None):
  import argparse
  database_replacement = "%s/.bob_bio_databases.txt" % os.environ["HOME"]

  parser = argparse.ArgumentParser(description="Prints a list of directories for registered databases", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-D', '--database-directories-file', metavar = 'FILE', default = database_replacement, help = 'The file, where database directories are stored (to avoid changing the database configurations)')

  args = parser.parse_args(command_line_parameters)

  # get registered databases
  databases = bob.bio.base.utils.resources.database_directories(replacements=args.database_directories_file)

  # print directories for all databases
  for d in sorted(databases):
    print ("\n%s:" % d)

    print ("Original data: %s" % databases[d][0])
    if len(databases[d]) > 1:
      print ("Annotations: %s" % databases[d][1])
