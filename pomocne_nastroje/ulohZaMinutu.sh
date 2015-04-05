#!/bin/bash

pocetUlohVeFronte() {
   qstat | grep "$USER" | wc -l
}

echo "Jednou za minutu vypise statistiku z fronty qstat pro uzivatele $USER"
aktualni=`pocetUlohVeFronte`
sleep 60

while :; do
   clear;
   predchozi="$aktualni";
   aktualni=`pocetUlohVeFronte`
   zaMinutu=$((predchozi - aktualni ))
   echo "********************"
   echo "Vypocteno $zaMinutu uloh za minutu."
   echo "Ve fronte zbyva $aktualni uloh"
   echo "Zbyvajici cas cca $(
      if [ $zaMinutu -eq 0 ]; then
         echo "N/A"
      else
         minut=$((aktualni / zaMinutu)) 
         [ "$minut" -ge 60 ] && echo -n "$(( minut / 60)) hodin "
         echo $((minut % 60 ))
      fi ) minut"
   echo "********************"
   sleep 60
done

