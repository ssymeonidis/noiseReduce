#################################################################
# noiseReduce Tensorflow Project 
# Copyright (C) 2024 Simeon Symeonidis
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#################################################################


# process arguments
def parse(args):
  out = dict()
  for arg in args:
    vals = arg.split('=')
    if len(vals) != 2:
      print('warning: non parseable argument')
    out[vals[0]] = vals[1]
  return out

# command line interface
if __name__ == "__main__":
  import sys
  out = parse(sys.argv[1:])
  print(out)
