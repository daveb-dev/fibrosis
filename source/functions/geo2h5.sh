#
# 	Take a .geo gmsh file and get the mesh file in both .xml and .h5 formats
#
#	FG - 2016
#

# TODO: permitir que el input venga con extensión y separar string dentro del script

# Generate .msh mesh file with GMSH 
gmsh -2 $1.geo

# Convert .msh to xml dolfin format
dolfin-convert $1.msh $2.xml

# Convert .xml file to .h5 format for parallel processing
python xml2hdf5.py $2.xml

# Erase not useful files
echo "Deleting intermediate files..."
sudo rm $1.msh $2.xml $2_facet_region.xml $2_physical_region.xml