#!/bin/bash
USAGE="USAGE
   $0 [ jmeno_uzivatele ]

   Vypisuje statistiku provadenych uloh z fronty \`qstat\` pro 
   uzivatele urcenym jmeno_uzivatele. Pokud neni specifikovano,
   bere se hodnota promenne prostredi \$USER.
"

pocetUlohVeFronte() {
   qstat | awk -v u=^$1$ 'BEGIN { pocet = 0 } $4 ~ u { pocet++ } END { print pocet }'
}

uzivatel=${1-$USER}

pocetNaZacatku=`pocetUlohVeFronte "$uzivatel"`
casNaZacatku=`date +%s`

[ "$pocetNaZacatku" -eq 0 ] && {
   echo "Ve fronte nejsou zadne ulohy uzivatele ${uzivatel}. Koncim."
   exit 0
}

clear
echo "Vypisuje statistiku z fronty qstat pro uzivatele $uzivatel"
sleep 5

while :; do
   pocetTed=`pocetUlohVeFronte "$uzivatel"`
   casTed=`date +%s`
   rozdilPoctu=$(( pocetNaZacatku - pocetTed ))
   rozdilCasu=$(( casTed - casNaZacatku ))
   zaMinutu=`echo "scale=1; 60 * $rozdilPoctu / $rozdilCasu" | bc`

   [ "$rozdilPoctu" -lt 0 ] && {
      echo "Pocet uloh se navysil"
      exit 1
   }

   clear
   echo "************************************************************"
   echo "Vypocteno $zaMinutu uloh za minutu."
   echo "Ve fronte zbyva $pocetTed uloh"
   echo "Zbyvajici cas cca: $(
      if [ `echo "$zaMinutu == 0" | bc` -eq "1" ]; then
         echo "N/A"
      else
         minut=`echo "scale=0; $pocetTed / $zaMinutu" | bc`
         [ "$minut" -ge 60 ] && echo -n "$(( minut / 60)) hodin "
         echo $(( minut % 60 ))
      fi ) minut"
   echo "************************************************************"

   [ "$pocetTed" -eq 0 ] && {
      echo "Zadne dalsi ulohy ve fronte, koncim."
      exit 0 
   }
   sleep 10
done

