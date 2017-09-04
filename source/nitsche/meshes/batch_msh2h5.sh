#!/bin/sh
source fenics.dev

for file in "$@"
do
    echo "Processing file $file"
    dolfin-convert "$file"  "${file%.*}".xml
    python ../functions/xml2hdf5.py "${file%.*}".xml
    rm "${file%.*}".xml
    if [ -f "${file%.*}"_facet_region.xml ]; then
        rm "${file%.*}"_facet_region.xml 
    fi
    if [ -f "${file%.*}"_physical_region.xml ]; then
        rm "${file%.*}"_physical_region.xml
    fi
done
