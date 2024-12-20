# -*- coding: utf-8 -*-
#ultima actualizacion: 10-07-2014, funcion getSymmetricBundle()

#from soma import aims
import numpy as N
import os
import shutil
import math
from collections import defaultdict

def mesh_neighbors(polygons):
    #
    vertex_poly = defaultdict(set)

    for i, poly in enumerate(polygons):
        for ver in poly:
            vertex_poly[ver].add(i)
            
    poly_poly = defaultdict(set)
    
    for i, poly in enumerate(polygons):
        for ver in poly:
            poly_poly[i] = poly_poly[i].union(vertex_poly[ver])
            poly_poly[i].remove(i)
            
    for k, v in poly_poly.items():
        poly_poly[k] = list(v)
    
    return poly_poly

def read_mesh( infile ):
    
    f = open(infile, 'rb');
    
    f.read(17);
    dim = N.frombuffer( f.read( 4 ), 'i' )[ 0 ];
    f.read(8);
    
    total_vertex = N.frombuffer( f.read( 4 ), 'i' )[ 0 ];
    vertex = N.frombuffer( f.read( 4 * dim * total_vertex), 'f' ).reshape( -1, 3 );
    
    if total_vertex < 50000:
        f.read(8);
    
    else:
        f.read(8*99259);
        
    total_polygons = N.frombuffer( f.read( 4 ), 'i' )[ 0 ];
    polygons = N.frombuffer( f.read( 4 * dim * total_polygons), 'i' ).reshape( -1, 3 );
    f.close();
        
    return vertex, polygons;

def read_mesh_obj( infile ):
    vertex = []
    polygons = []
    len_vertex = 0
    len_polygons = 0
    
    with open(infile, 'r') as f:

        for l in f:
            e = l.split()
            
            if len(e) == 0:
                continue
            else:
                ID = e[0]
            
            if ID == 'v':
                vertex.append(e[1:])
                len_vertex += 1
                ID = 'n'
                
            elif ID == 'f':
                polygons.append([int(e[1].split('//')[0])-1, int(e[2].split('//')[0])-1, int(e[3].split('//')[0])-1])
                len_polygons += 1
                ID = 'n'
                
    vertex = N.asarray(vertex, dtype = N.float32)
    polygons = N.asarray(polygons, dtype = N.int)
    
    return vertex, polygons
    

def getBundleNames( bundlefile ):

  #get center names from bundle file
  ns = dict()
  execfile( bundlefile, ns )
  bunlist = ns[ 'attributes' ][ 'bundles' ]
  centers_num = len( bunlist ) / 2
  labels = []
  for i in range( centers_num ):

    ind = i * 2
    labels.append( bunlist[ ind ] )
  return labels

def getBundleNamesAndSizes( bundlefile ):

  #get center names from bundle file
  ns = dict()
  execfile( bundlefile, ns )
  bunlist = ns[ 'attributes' ][ 'bundles' ]
  curves_count = ns[ 'attributes' ][ 'curves_count' ]
  centers_num = len( bunlist ) / 2
  labels = []
  sizes = []
  for i in range( centers_num ):

    ind = i * 2
    labels.append( bunlist[ ind ] )
  prec_size = bunlist[ 1 ]
  for i in range( centers_num - 1 ):

    ind = i * 2 + 3
    prec_size_tmp = bunlist[ ind ]
    sizes.append( prec_size_tmp - prec_size )
    prec_size = prec_size_tmp
  sizes.append( curves_count - bunlist[ len( bunlist ) - 1 ] )
  return labels, sizes, curves_count

def getBundleSize( bundlefile ):

  #get center names from bundle file
  ns = dict()
  execfile( bundlefile, ns )
  size = ns[ 'attributes' ][ 'curves_count' ]
  return  size

def getBundleNb( bundlefile ):

  #get center names from bundle file
  ns = dict()
  execfile( bundlefile, ns )
  bunlist = ns[ 'attributes' ][ 'bundles' ]
  return  len( bunlist ) / 2

def allFibersToOneBundle( bundlefile, bundlename = '1', mode = 0 ):
	
  ns = dict()
  execfile( bundlefile, ns )
  curves_count = ns[ 'attributes' ][ 'curves_count' ]
  if ( mode == 1 ):

    bunlist = ns[ 'attributes' ][ 'bundles' ]
    bundlename = bunlist[ 0 ]
  bundles = '[ \'' + bundlename + '\', 0 ]'
  # write minf file
  minf = """attributes = {\n    'binary' : 1,\n    'bundles' : %s,\n    'byte_order' : 'DCBA',\n    'curves_count' : %s,\n    'data_file_name' : '*.bundlesdata',\n    'format' : 'bundles_1.0',\n    'space_dimension' : 3\n  }"""
  open( bundlefile, 'w' ).write( minf % ( bundles, curves_count ) )

def changeBundleNameToNumber( bundlefile, bundleout, offset = 0 ):
	
  ns = dict()
  execfile( bundlefile, ns )
  curves_count = ns[ 'attributes' ][ 'curves_count' ]
  bunlist = ns[ 'attributes' ][ 'bundles' ]
  # write minf file
  minf = """attributes = {\n    'binary' : 1,\n    'bundles' : %s,\n    'byte_order' : 'DCBA',\n    'curves_count' : %s,\n    'data_file_name' : '*.bundlesdata',\n    'format' : 'bundles_1.0',\n    'space_dimension' : 3\n  }"""
	
  for i in range( len( bunlist ) / 2 ):
    ind = i * 2
    bunlist[ ind ] = str( i + offset )
    #print i, ' : len= ', len(points[i])

  bundlesstr = '['
  l = len( bunlist ) / 2
  for i in range( l - 1 ):
    ind = i * 2
    bundlesstr += ' \'' + bunlist[ ind ] + '\', ' \
                        + str( bunlist[ ind + 1 ] ) + ','

  bundlesstr += ' \'' + bunlist[ ind + 2 ] + '\', ' \
                      + str( bunlist[ ind + 3 ] ) + ' ]'
  open( bundleout, 'w' ).write( minf % ( bundlesstr, curves_count ) )
  shutil.copyfile( bundlefile + 'data',  bundleout + 'data' )

def oneFiberPerBundle( bundlefile, bundleout, offset = 0 ):
  ns = dict()
  execfile( bundlefile, ns )
  curves_count = ns[ 'attributes' ][ 'curves_count' ]
  bunlist = ns[ 'attributes' ][ 'bundles' ]
  # write minf file
  minf = """attributes = {\n    'binary' : 1,\n    'bundles' : %s,\n    'byte_order' : 'DCBA',\n    'curves_count' : %s,\n    'data_file_name' : '*.bundlesdata',\n    'format' : 'bundles_1.0',\n    'space_dimension' : 3\n  }"""
  bunlist2 = []
  for i in range( curves_count ):

    bunlist2.append( str( i + offset ) )
    bunlist2.append( i )
  bundlesstr = '['
  l = len( bunlist2 ) / 2
  for i in range( l - 1 ):
    ind = i * 2
    bundlesstr += ' \'' + bunlist2[ ind ] + '\', '  \
                        + str(bunlist2[ ind + 1 ] ) + ','

  bundlesstr += ' \'' + bunlist2[ ind + 2 ] + '\', ' \
                      + str( bunlist2[ ind + 3 ] ) + ' ]'
  open( bundleout, 'w' ).write( minf % ( bundlesstr, curves_count ) )
  shutil.copyfile( bundlefile + 'data',  bundleout + 'data')


def read_bundle( infile ):

  points = []
  bun_file = infile + 'data'
  os.path.getsize( bun_file )
  bytes = os.path.getsize( bun_file )
  num = bytes / 4

  num_count = 0
  f = open( bun_file , 'rb')
  while num_count < num:

    p = N.frombuffer( f.read( 4 ), 'i' )[ 0 ] # numero de puntos de la fibra
    vertex = N.frombuffer( f.read( p * 3 * 4 ), 'f' ).reshape( -1, 3 ) # lee coordenadas fibra
    points.append( vertex )
    num_count = num_count + 1 + ( p * 3 )

  f.close()

  return points#, bundles

def read_OneFiber( infile ):

  points = []
  bun_file = infile + 'data'
  os.path.getsize( bun_file )
  bytes = os.path.getsize( bun_file )
  num = bytes / 4
  f = open( bun_file )
  p = N.frombuffer( f.read( 4 ), 'i' )[ 0 ]
  vertex = N.frombuffer( f.read( p * 3 * 4 ), 'f' ).reshape( -1, 3 )
  points.append( vertex )
  f.close()

  #bundles = []
  return points#, bundles

def write_bundle( outfile, points ):

  #write bundles file
  f = open( outfile + 'data','w' )
  ncount = len( points )
  for i in range( ncount ):
    f.write(N.array( [ len( points[ i ] ) ], N.int32 ).tostring() )
    f.write( points[ i ].ravel().tostring() )

  f.close()

  # wrtie minf file
  minf = """attributes = {\n    'binary' : 1,\n    'bundles' : %s,\n    'byte_order' : 'DCBA',\n    'curves_count' : %s,\n    'data_file_name' : '*.bundlesdata',\n    'format' : 'bundles_1.0',\n    'space_dimension' : 3\n  }"""
  open( outfile, 'w' ).write(minf % ( [ 'points', 0 ], ncount ) )

def read_bundle_severalbundles( infile ):

  points = []
  bun_file = infile + 'data'
  os.path.getsize( bun_file )
  bytes = os.path.getsize( bun_file )
  num = bytes / 4

  ns = dict()
  execfile( infile, ns )
  bundlescount = ns[ 'attributes' ][ 'bundles' ]
  curvescount = ns[ 'attributes' ][ 'curves_count' ]
  bunnames = []
  bunstart = []
  bun_num = len( bundlescount ) / 2
  count = 0
  for i in range( bun_num ):

    bunnames.append( bundlescount[ count ] )
    count = count + 1
    bunstart.append( bundlescount[ count ] )
    count = count + 1
    points.append( [] )

  bun_size = []
  if len( bunstart ) > 1:

    for b in range( len( bunstart ) - 1 ):

      bun_size.append( bunstart[ b + 1 ] - bunstart[ b ] )
    bun_size.append( curvescount - bunstart[ b + 1 ] )
  else:

    bun_size.append( curvescount )	
  num_count = 0
  f = open( bun_file )
  bun_count = 0
  num_count_bundle = 0
  while num_count < num:

    p = N.frombuffer( f.read( 4 ), 'i' )[ 0 ]
    vertex = N.frombuffer( f.read( p * 3 * 4 ), 'f' ).reshape( -1, 3 )
    points[ bun_count ].append( vertex )
    #print num_count, p, bun_count, num_count_bundle
    num_count_bundle = num_count_bundle + 1
    if num_count_bundle == bun_size[ bun_count ]:
      bun_count = bun_count + 1
      num_count_bundle = 0
    num_count = num_count + 1 + ( p * 3 )

  f.close()
  return points, bunnames


def write_bundle_severalbundles( outfile, points, bundles = [] ):

  #write bundles file
  f = open( outfile + 'data','w' )
  ncount = 0
  for i in range( len( points ) ):

    size = len( points[ i ] )
    ncount = ncount + size
    bun = points[ i ]
    for i in range( size ):

      f.write( N.array( [ len( bun[ i ] ) ], N.int32 ).tostring() )
      f.write( bun[ i ].ravel().tostring() )
  f.close()
  # wrtie minf file
  minf = """attributes = {\n    'binary' : 1,\n    'bundles' : %s,\n    'byte_order' : 'DCBA',\n    'curves_count' : %s,\n    'data_file_name' : '*.bundlesdata',\n    'format' : 'bundles_1.0',\n    'space_dimension' : 3\n  }"""

  bundles_list = []
  ind = 0
  for i in range( len( points ) ):

    if bundles == []:

      bundles_list.append( str( i ) )
    else:

      bundles_list.append( bundles[ i ] )
    bundles_list.append( ind )
    #print i, ' : len= ', len(points[i])
    ind = ind + len( points[ i ] )

  bundlesstr = '['
  l = len( bundles_list ) / 2
  for i in range( l - 1 ):

    ind = i*2
    bundlesstr += ' \'' + bundles_list[ ind ] + '\', ' \
                        + str( bundles_list[ ind + 1 ] ) + ','
  bundlesstr += ' \'' + bundles_list[ ind + 2 ] + '\', ' \
                      + str( bundles_list[ ind + 3 ] ) + ' ]'

  open( outfile, 'w' ).write( minf % ( bundlesstr, ncount ) )


def getMinAndMaxFiberSteps(bunfile, returnDistances = False):
  points = read_bundle(bunfile)
  minv = 10000
  maxv = 0
  dists = []
  for p in points:
    for i in range(len(p)-1):
      p1 = p[i]
      p2 = p[i+1]
      x = p1[0]-p2[0]
      y = p1[1]-p2[1]
      z = p1[2]-p2[2]
      d = x*x + y*y + z*z
      if returnDistances:
        dists.append(math.sqrt(d))
      if d < minv:
        minv = d
      if d > maxv:
        maxv = d
  minv = math.sqrt(minv)
  maxv = math.sqrt(maxv)
  return minv, maxv, dists

def getSymmetricBundle(bunfile, bunout, onebundle_name = None):
  #if onebundle_name != None, all fibers are put into one bundle
  #named 'onebundle_name'
  points, bunnames = read_bundle_severalbundles(bunfile)
  points2 = []
  for bundle in points:
    bun = []
    for fiber in bundle:
      fiber2 = N.array(fiber)
      for p in fiber2:
        p[0] = -p[0]
      bun.append(fiber2)
    points2.append(bun)
  write_bundle_severalbundles(bunout, points2, bunnames)
  if onebundle_name != None:
    allFibersToOneBundle(bunout, onebundle_name)
  return

def read_intersection( infile ):
    
    f = open(infile, 'rb');

    total_triangles = N.frombuffer( f.read( 4 ), N.uint32 )[ 0 ];
    
    InTri = N.frombuffer( f.read( 4 * total_triangles), N.uint32 );
    FnTri = N.frombuffer( f.read( 4 * total_triangles), N.uint32 );
    
    InPoints = N.frombuffer( f.read( 4 * 3 * total_triangles), N.float32 ).reshape(-1,3);
    FnPoints = N.frombuffer( f.read( 4 * 3 * total_triangles), N.float32 ).reshape(-1,3);
    
    fib_index = N.frombuffer( f.read( 4 * total_triangles), N.uint32 );
    
    f.close();
    return InTri, FnTri, InPoints, FnPoints, fib_index;

