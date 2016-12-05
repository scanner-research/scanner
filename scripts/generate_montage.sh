SCANNER=/home/wcrichto/scanner
FG=/home/wcrichto/mp/film_grammar_lite/python/fg_pipeline

cd $SCANNER
rm -f /tmp/vid.txt
echo $1 > /tmp/vid.txt
time ./build/scanner_server ingest video tmp /tmp/vid.txt
time ./build/scanner_server run tmp base histogram hist --pus_per_node=2
time ./build/scanner_server run tmp base median med --pus_per_node=8
time python python/decode.py tmp hist
mv -f tmp_histograms.npy $FG/tmp/histograms.npy
cd $FG
time python process_movie.py tmp
mv -f tmp/shots.txt /bigdata/wcrichto/shots/tmp.txt
cd $SCANNER
time python python/movie_graphs.py
rm -f /bigdata/wcrichto/shots/tmp.txt
./build/scanner_server rm dataset tmp
