# skript pro gnuplot

reset
adresar = "../../mereni/"

# podadresarDynamicDijkstraHuste = adresar."dynamic_huste_dijkstra/"."vysledky/"
# podadresarDynamicDijkstraRidke = adresar."dynamic_ridke_dijkstra/"."vysledky/"
# podadresarDynamicFloydHuste    = adresar."dynamic_huste_floyd-warshall/"."vysledky/"
# podadresarDynamicFloydRidke    = adresar."dynamic_ridke_floyd-warshall/"."vysledky/"
podadresarStaticDijkstraHuste  = adresar."static_huste_dijkstra/"."vysledky/"
podadresarStaticDijkstraRidke  = adresar."static_ridke_dijkstra/"."vysledky/"
podadresarStaticFloydHuste     = adresar."static_huste_floyd-warshall_v2/"."vysledky/"
podadresarStaticFloydRidke     = adresar."static_ridke_floyd-warshall_v2/"."vysledky/"
# podadresarStaticFloydHuste     = adresar."static_huste_floyd-warshall/"."vysledky/"
# podadresarStaticFloydRidke     = adresar."static_ridke_floyd-warshall/"."vysledky/"
# podadresarStaticFloydHuste     = adresar."static_huste_vzdalenosti_floyd-warshall/"."vysledky/"
# podadresarStaticFloydRidke     = adresar."static_ridke_vzdalenosti_floyd-warshall/"."vysledky/"

adresarPorovnani                     = adresar."porovnani/"
podadresarPorovnaniDijkstraHustota   = adresarPorovnani."dijkstra_static_huste-ridke/"
podadresarPorovnaniFloydHustota      = adresarPorovnani."floyd-warshall_static_huste-ridke/"
podadresarPorovnaniDijkstraPlanovani = adresarPorovnani."dijkstra_huste_static-dynamic/"
podadresarPorovnaniFloydPlanovani    = adresarPorovnani."floyd-warshall_huste_static-dynamic/"

set terminal pngcairo size 600,400 enhanced font 'Helvetica,9'

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

# ======================================================================
# rychlost vypoctu v zavislosti na poctu vlaken
# ======================================================================

set key top right

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

set key top left

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

set key bottom left

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


# ======================================================================
# Porovnani ridke huste grafy 
# ======================================================================

set key top left 

set ylabel 'Pomer casu huste/ridke'
set yrange [0.8:1.2]

nadpis    = "Porovnani pomeru rychlosti vypoctu v zavislosti na hustote grafu"
podnadpis = "Algoritmus Dijsktra, staticke planovani"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarPorovnaniDijkstraHustota
set output "02-04-Dijsktra_hustota.png"

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


# ======================================================================
# Porovnani dynamicke, staticke planovani
# ======================================================================

set key top left 

set ylabel 'Pomer casu staticke/dynamicke'
set yrange [0.8:1.2]

nadpis    = "Porovnani pomeru rychlosti vypoctu v zavislosti na planovani uloh"
podnadpis = "Algoritmus Dijsktra, huste grafy"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarPorovnaniDijkstraPlanovani
set output "02-05-Dijsktra_schedule.png"

plot \
   1 with lines linecolor "black" title "y = 1", \
   podadresar."vysledky-1000.txt" using 1:2 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 1:2 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 1:2 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 1:2 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 1:2 linestyle 51 title "n = 5000";


podnadpis = "Algoritmus Floyd-Warshall, huste grafy"
set title nadpis."\n{/*0.8 ".podnadpis."}"

podadresar = podadresarPorovnaniFloydHustota
set output "02-05-Floyd_schedule.png"

plot \
   1 with lines linecolor "black" title "y = 1", \
   podadresar."vysledky-1000.txt" using 1:2 linestyle 11 title "n = 1000", \
   podadresar."vysledky-2000.txt" using 1:2 linestyle 21 title "n = 2000", \
   podadresar."vysledky-3000.txt" using 1:2 linestyle 31 title "n = 3000", \
   podadresar."vysledky-4000.txt" using 1:2 linestyle 41 title "n = 4000", \
   podadresar."vysledky-5000.txt" using 1:2 linestyle 51 title "n = 5000";



