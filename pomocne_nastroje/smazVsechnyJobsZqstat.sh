#/bin/bash

ids=`qstat | awk -v "u=\\\\\<$USER\\\\\>" '
   ( $0 ~ u ) && ( $1 ~ /^[0-9]+$/ ) {
      print $1
   }'`

for id in $ids; do
   qdel "$id"
done

