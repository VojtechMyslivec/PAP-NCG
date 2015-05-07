# skript pro gnuplot

reset
adresar = "../../mereni/03_cuda/"

podadresarDijkstraHuste  = adresar."cuda_huste_dijkstra_v1/"."vysledky/"
podadresarDijkstraRidke  = adresar."cuda_ridke_dijkstra_v1/"."vysledky/"
podadresarFloydHusteV1   = adresar."cuda_huste_floyd-warshall_v1/"."vysledky/"
podadresarFloydRidkeV1   = adresar."cuda_ridke_floyd-warshall_v1/"."vysledky/"
podadresarFloydHusteV2   = adresar."cuda_huste_floyd-warshall_v2/"."vysledky/"

# adresarPorovnani                     = adresar."porovnani/"
# podadresarPorovnaniDijkstraHustota   = adresarPorovnani."dijkstra_static_huste-ridke/"
# podadresarPorovnaniFloydHustota      = adresarPorovnani."floyd-warshall_static_huste-ridke/"
# podadresarPorovnaniDijkstraPlanovani = adresarPorovnani."dijkstra_huste_static-dynamic/"
# podadresarPorovnaniFloydPlanovani    = adresarPorovnani."floyd-warshall_huste_static-dynamic/"

set terminal pngcairo size 600,450 enhanced font 'Helvetica,9'

set key box 0

# osy
# odstrani ramecek nahore a vpravo
set style line 91 linecolor rgb '#000000' linetype 1 linewidth 2
set border 3 back linestyle 91
set tics nomirror

# mrizka
set style line 92 linecolor rgb '#808080' linetype 0 linewidth 1
set grid back linestyle 92

# format car
set style line 01 linecolor rgb '#8b8b00' linetype 5 linewidth 2
set style line 11 linecolor rgb '#8b1a0e' linetype 1 linewidth 4
set style line 12 linecolor rgb '#aa4020' linetype 1 linewidth 4
set style line 13 linecolor rgb '#191970' linetype 1 linewidth 3
set style line 21 linecolor rgb '#5e9c36' linetype 2 linewidth 4
set style line 22 linecolor rgb '#20aa40' linetype 2 linewidth 4
set style line 23 linecolor rgb '#FFB90F' linetype 2 linewidth 3
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


set xrange [28:1024]
set xtics ( 32, 64, 128, 256, 512, 768, 1024 )
set logscale x 2

# ======================================================================
# rychlost vypoctu v zavislosti na poctu vlaken v bloku
# ======================================================================

set key outside bottom center horizontal

set xlabel 'Pocet vlaken v bloku'

set ylabel 'Cas [s]'
set yrange [0:300]

nadpis    = "Prumerny cas vypoctu v zavislosti na poctu vlaken v bloku"
podnadpis = "Algoritmus Dijkstra"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarDijkstraHuste
set output "03-01-Dijkstra_cas.png"

plot \
   podadresar."vysledky-1000.txt" using 2:4 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 2:4 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 2:4 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 2:4 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 2:4 linestyle 51 title "n = 5000";

podnadpis = "Algoritmus Floyd-Warshall"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarFloydHusteV1
set output "03-01-Floyd_v1_cas.png"

plot \
   podadresar."vysledky-1000.txt" using 2:4 linestyle 12 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 2:4 linestyle 22 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 2:4 linestyle 32 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 2:4 linestyle 42 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 2:4 linestyle 52 title "n = 5000";

# zoom 1
set yrange [0:30]
set output "03-01-Floyd_v1_cas_zoom.png"
replot

podnadpis = "Algoritmus Floyd-Warshall shared \\\& staged"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarFloydHusteV2
set output "03-01-Floyd_v2_cas.png"

plot \
   podadresar."vysledky-1000.txt" using 2:4 linestyle 12 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 2:4 linestyle 22 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 2:4 linestyle 32 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 2:4 linestyle 42 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 2:4 linestyle 52 title "n = 5000";

# zoom 2
set yrange [0:7]
set output "03-01-Floyd_v2_cas_zoom.png"
replot

# ======================================================================
# Zrychleni vypoctu
# ======================================================================

set ylabel 'Zrychleni'
set yrange [0:10]

nadpis    = "Zrychleni vypoctu v zavislosti na poctu vlaken v bloku"
podnadpis = "Algoritmus Dijkstra"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarDijkstraHuste
set output "03-02-Dijkstra_zrychleni.png"

plot \
   podadresar."vysledky-1000.txt" using 2:5 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 2:5 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 2:5 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 2:5 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 2:5 linestyle 51 title "n = 5000";

# unzoom 1
set yrange [0:40]
set output "03-02-Dijkstra_zrychleni_unzoom.png"
replot

podnadpis = "Algoritmus Floyd-Warshall"

set title nadpis."\n{/*0.8 ".podnadpis."}"
podadresar = podadresarFloydHusteV1
set output "03-02-Floyd_v1_zrychleni.png"

plot \
   podadresar."vysledky-1000.txt" using 2:5 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 2:5 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 2:5 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 2:5 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 2:5 linestyle 51 title "n = 5000";

# unzoom 2
set yrange [0:100]
set output "03-02-Floyd_v1_zrychleni_unzoom.png"
replot

podnadpis = "Algoritmus Floyd-Warshall shared \\\& staged"

set title nadpis."\n{/*0.8 ".podnadpis."}"
podadresar = podadresarFloydHusteV2
set output "03-02-Floyd_v2_zrychleni.png"

plot \
   podadresar."vysledky-1000.txt" using 2:5 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 2:5 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 2:5 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 2:5 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 2:5 linestyle 51 title "n = 5000";

exit
# TBA
# ======================================================================
# Porovnani ridke huste grafy 
# ======================================================================

set key top left 

set ylabel 'Pomer casu huste/ridke'
set yrange [0.8:1.2]

nadpis    = "Porovnani pomeru rychlosti vypoctu v zavislosti na hustote grafu"
podnadpis = "Algoritmus Dijkstra, staticke planovani"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarPorovnaniDijkstraHustota
set output "02-04-Dijkstra_hustota.png"

plot \
   1 with lines linecolor "black" title "y = 1", \
   podadresar."vysledky-1000.txt" using 1:2 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 1:2 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 1:2 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 1:2 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 1:2 linestyle 51 title "n = 5000";

podnadpis = "Algoritmus Floyd-Warshall, staticke planovani"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarPorovnaniFloydHustota
set output "02-04-Floyd_hustota.png"

plot \
   1 with lines linecolor "black" title "y = 1", \
   podadresar."vysledky-1000.txt" using 1:2 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 1:2 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 1:2 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 1:2 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 1:2 linestyle 51 title "n = 5000";


