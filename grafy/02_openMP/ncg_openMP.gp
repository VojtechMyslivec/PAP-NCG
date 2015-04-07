# skript pro gnuplot

reset
adresar = "../../mereni/"
podadresarDynamicDijkstraHuste = adresar."dynamic_huste_dijkstra_2015-04-05_10-18-04/"."vysledky/"
podadresarDynamicDijkstraRidke = adresar."dynamic_ridke_dijkstra_2015-04-05_10-18-19/"."vysledky/"
podadresarDynamicFloydHuste    = adresar."dynamic_huste_floyd-warshall_2015-04-05_10-19-06/"."vysledky/"
podadresarDynamicFloydRidke    = adresar."dynamic_ridke_floyd-warshall_2015-04-05_10-18-45/"."vysledky/"
podadresarStaticDijkstraHuste  = adresar."static_huste_dijkstra_2015-03-29_23-07-08/"."vysledky/"
podadresarStaticDijkstraRidke  = adresar."static_ridke_dijkstra_2015-03-30_20-34-20/"."vysledky/"
podadresarStaticFloydHuste     = adresar."static_huste_floyd-warshall_2015-03-29_23-07-20/"."vysledky/"
podadresarStaticFloydRidke     = adresar."static_ridke_floyd-warshall_2015-03-30_20-34-33/"."vysledky/"


set terminal pngcairo size 600,400 enhanced font 'Helvetica,9'

# osy
# odstrani ramecek nahore a vpravo
set style line 91 linecolor rgb '#000000' linetype 1 linewidth 2
set border 3 back linestyle 91
set tics nomirror

set key top right box 0
# set key outside bottom center

# mrizka
set style line 92 linecolor rgb '#808080' linetype 0 linewidth 1
set grid back linestyle 92

# format car
set style line 01 linecolor rgb '#8b8b00' linetype 5 linewidth 2
set style line 11 linecolor rgb '#8b1a0e' linetype 1 linewidth 4
set style line 12 linecolor rgb '#aa4020' linetype 1 linewidth 4
set style line 21 linecolor rgb '#5e9c36' linetype 2 linewidth 4
set style line 22 linecolor rgb '#20aa40' linetype 2 linewidth 4
set style line 31 linecolor rgb '#191970' linetype 3 linewidth 2
set style line 32 linecolor rgb '#3D59AB' linetype 3 linewidth 2
set style line 33 linecolor rgb '#0000FF' linetype 3 linewidth 2
set style line 41 linecolor rgb '#8B0000' linetype 6 linewidth 2
set style line 42 linecolor rgb '#CD2626' linetype 6 linewidth 2
set style line 43 linecolor rgb '#ff0000' linetype 6 linewidth 2
set style line 51 linecolor rgb '#FFB90F' linetype 4 linewidth 2
set style line 52 linecolor rgb '#B8860B' linetype 4 linewidth 2
set style line 53 linecolor rgb '#ffff00' linetype 4 linewidth 2
set style line 61 linecolor rgb '#000000' linetype 8 linewidth 2
set style line 62 linecolor rgb '#303030' linetype 8 linewidth 2
set style line 63 linecolor rgb '#666666' linetype 8 linewidth 2

# ======================================================================
# rychlost vypoctu v zavislosti na poctu vlaken
# ======================================================================

set xrange [0:25]
set xlabel 'Pocet vlaken'

set ylabel 'Cas [s]'
set yrange [0:450]

nadpis    = "Prumerny cas vypoctu v zavislosti na poctu vlaken"
podnadpis = "Algoritmus Dijsktra, huste grafy, staticke planovani"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarStaticDijkstraHuste
set output "02-01-Dijsktra_cas.png"

plot \
   podadresar."vysledky-1000.txt" using 1:2 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 1:2 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 1:2 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 1:2 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 1:2 linestyle 51 title "n = 5000";

podnadpis = "Algoritmus Floyd-Warshall, huste grafy, staticke planovani"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarStaticFloydHuste
set output "02-01-Floyd_cas.png"

plot \
   podadresar."vysledky-1000.txt" using 1:2 linestyle 12 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 1:2 linestyle 22 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 1:2 linestyle 32 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 1:2 linestyle 42 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 1:2 linestyle 52 title "n = 5000";

# ======================================================================
# Zrychleni vypoctu
# ======================================================================

set key left

set ylabel 'Zrychleni'
set yrange [0:16]

nadpis    = "Zrychleni vypoctu v zavislosti na poctu vlaken"
podnadpis = "Algoritmus Dijsktra, huste grafy, staticke planovani"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarStaticDijkstraHuste
set output "02-02-Dijsktra_zrychleni.png"

plot \
   x with lines linecolor "black" title "y = x", \
   podadresar."vysledky-1000.txt" using 1:3 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 1:3 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 1:3 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 1:3 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 1:3 linestyle 51 title "n = 5000";

podnadpis = "Algoritmus Floyd-Warshall, huste grafy, staticke planovani"

set title nadpis."\n{/*0.8 ".podnadpis."}"
podadresar = podadresarStaticFloydHuste
set output "02-02-Floyd_zrychleni.png"

plot \
   x with lines linecolor "black" title "y = x", \
   podadresar."vysledky-1000.txt" using 1:3 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 1:3 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 1:3 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 1:3 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 1:3 linestyle 51 title "n = 5000";

# ======================================================================
# Efektivita
# ======================================================================

set key bottom

set ylabel 'Efektivita'
set yrange [0:1]

nadpis    = "Efektivita vypoctu v zavislosti na poctu vlaken"
podnadpis = "Algoritmus Dijsktra, huste grafy, staticke planovani"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarStaticDijkstraHuste
set output "02-03-Dijsktra_efektivita.png"

plot \
   1 with lines linecolor "black" title "y = 1", \
   podadresar."vysledky-1000.txt" using 1:4 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 1:4 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 1:4 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 1:4 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 1:4 linestyle 51 title "n = 5000";


podnadpis = "Algoritmus Floyd-Warshall, huste grafy, staticke planovani"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarStaticFloydHuste
set output "02-03-Floyd_efektivita.png"

plot \
   1 with lines linecolor "black" title "y = 1", \
   podadresar."vysledky-1000.txt" using 1:4 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 1:4 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 1:4 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 1:4 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 1:4 linestyle 51 title "n = 5000";


