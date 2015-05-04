"""Prints a detailed list of all resources that are registered, including the modules, where they have been registered."""

from __future__ import print_function
import bob.bio.base

def main():

  import argparse
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--details", '-d', nargs = '+',
                      choices = ('d', 'database', 'p', 'preprocessor', 'e', 'extractor', 'a', 'algorithm', 'g', 'grid'),
                      default = ('d', 'p', 'e', 'a', 'g'),
                      help = "Select the resource types that should be listed.")

  args = parser.parse_args()

  if 'd' in args.details or 'database' in args.details:
    print ("\nList of registered databases:")
    print (bob.bio.base.list_resources('database'))

  if 'p' in args.details or 'preprocessor' in args.details:
    print ("\nList of registered preprocessors:")
    print (bob.bio.base.list_resources('preprocessor'))

  if 'e' in args.details or 'extractor' in args.details:
    print ("\nList of registered extractors:")
    print (bob.bio.base.list_resources('extractor'))

  if 'a' in args.details or 'algorithm' in args.details:
    print ("\nList of registered algorithms:")
    print (bob.bio.base.list_resources('algorithm'))

  if 'g' in args.details or 'grid' in args.details:
    print ("\nList of registered grid configurations:")
    print (bob.bio.base.list_resources('grid'))

  print()
