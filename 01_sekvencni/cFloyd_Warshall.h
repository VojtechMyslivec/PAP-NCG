/* 
 * File:   cFloyd_Warshall.h
 * Author: novy
 *
 * Created on 2. březen 2015, 7:15
 */

#ifndef CFLOYD_WARSHALL_H
#define	CFLOYD_WARSHALL_H

#include "funkceSpolecne.h"
#include <iomanip>
#define FW_NEKONECNO       UNSIGNED_NEKONECNO
#define FW_NEDEFINOVANO    UINT_MAX

class cFloyd_Warshall {
public:
  //inicializuje a nastavi vnitrni promenne tridy jako matice pocetUlzu^2
  cFloyd_Warshall(unsigned ** graf, unsigned pocetUzlu);
  cFloyd_Warshall(const cFloyd_Warshall& orig);
  virtual ~cFloyd_Warshall();
  
  // vypocte nejkratsi cesty od vsech uzlu ke vsem ostatnim
  // slozitost vypoctu je O( |U|^3 )
  // (cesty od vsech uzlu do kazdheo uzlu pres jakykoliv uzel)
  void spustVypocet();
  
  // prohodi dva pointery mezi sebou pres pomocny pointer
  // slozitost je O(1)
  void prohodPredchoziAAktualni();
  
  // vypise vysledek po startovnich uzlech ve tvaru
  // id uzlu  X, uzly, jejich vzdalenosti a predchudci
  void vypisVysledekPoUzlech() const;
  
  // vypise vysledek jako matici vzdalenosti a matici predchudcu
  void vypisVysledekMaticove() const;
  
private:
  unsigned ** delkaPredchozi;
  unsigned ** delkaAktualni;
  unsigned ** predchudcePredchozi;
  unsigned ** predchudceAktualni;
  unsigned    pocetUzlu;
  
};

#endif	/* CFLOYD_WARSHALL_H */

