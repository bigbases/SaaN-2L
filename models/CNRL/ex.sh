#! /bin/sh

for files in /home/foursquare_community/*
do 
   python3 lda_link.py --input ${files} --reset 1 --cene 2 --fold 20 --cenecmty 10 --number-walks 10 --window-size 10 --walk-length 80 --veclen 256
done

