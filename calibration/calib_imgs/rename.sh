#!/bin/bash

i=1
for file in *.jpg; do
  if [ -f "$file" ]; then
    mv "$file" "img_$i.jpg"
    let i++
  fi
done
