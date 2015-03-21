/** cDijkstra.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 sekvencni cast
 *                 trida cDijkstra pro Dijkstruv algoritmus
 *                    upraven vypocet prioritni fronty -- misto haldy for cylky, 
 *                    aby slo paralelizovat
 *
 *
 */

#include "cDijkstra.h"

#include <iostream>
#include <iomanip>

using namespace std;
// ============================================================================

cDijkstra::cDijkstra( unsigned ** graf, unsigned pocetUzlu ) {
    // TODO melka kopie
    this->graf = graf;
    this->pocetUzlu = pocetUzlu;

    vzdalenost = new unsigned[pocetUzlu];
    predchudce = new unsigned[pocetUzlu];
    uzavreny   = new bool[pocetUzlu];
    pocetUzavrenychUzlu = 0;

    idInstance = 0;
    vzdalenostM = new unsigned*[pocetUzlu];
    predchudceM = new unsigned*[pocetUzlu];
    for ( unsigned i = 0; i < pocetUzlu; i++ ) {
        vzdalenostM[i] = new unsigned[pocetUzlu];
        predchudceM[i] = new unsigned[pocetUzlu];
        for ( unsigned j = 0; j < pocetUzlu; j++ ) {
           vzdalenostM[i][j] = DIJKSTRA_NEKONECNO;
           predchudceM[i][j] = DIJKSTRA_NEDEFINOVANO;
        }

    }
}

cDijkstra::~cDijkstra() {
    if (vzdalenost != NULL)
        delete [] vzdalenost;
    if (predchudce != NULL)
        delete [] predchudce;
    if (uzavreny != NULL)
        delete [] uzavreny;

    for (unsigned i = 0; i < pocetUzlu; i++) {
        delete [] vzdalenostM[i];
        delete [] predchudceM[i];
    }
    delete [] vzdalenostM;
    delete [] predchudceM;
}

bool cDijkstra::spustVypocet( ) {
   bool returnFlag = true;
   for ( unsigned idUzlu = 0 ; idUzlu < pocetUzlu ; idUzlu++ ) {

#ifdef DEBUG
      cout << "\nDijkstra pro uzel id = " << idUzlu << endl;      
#endif // DEBUG
      
      if ( vypoctiProUzel( idUzlu ) != true ) {
         cerr << "problem s vypoctem pro id = " << idUzlu<< endl;
         returnFlag = false;
      }
      else {
         // zkopiruje vysledek do matice
         for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
            vzdalenostM[idUzlu][i] = vzdalenost[i];
            predchudceM[idUzlu][i] = predchudce[i];
         }
      }

   }
   return returnFlag;
}

bool cDijkstra::vypoctiProUzel( unsigned idVychozihoUzlu ) {
    unsigned idUzlu;
    unsigned vzdalenostUzlu, vzdalenostSouseda, vzdalenostHrany, novaVzdalenost;
    idInstance = idVychozihoUzlu;
    
    if ( inicializace( idVychozihoUzlu ) != true ) {
        return false;
    }

#ifdef DEBUG
        vypisFrontu( );
#endif // DEBUG

    // dokud nenavstivi vsechny uzly
    while ( pocetUzavrenychUzlu < pocetUzlu ) {
        // uzly navstevuje podle nejmensi vzdalenosti
        if ( ! vyberMinimumZFronty( idUzlu ) ) 
           return false;
        uzavreny[idUzlu] = true;
        pocetUzavrenychUzlu++;
#ifdef DEBUG
        cerr << "\nzpracovavam uzel " << idUzlu << endl;
        vypisFrontu( );
#endif // DEBUG
        vzdalenostUzlu = vzdalenost[idUzlu];
        // pokud je stale nekonecna, znamena to nedostupny uzel;
        if ( vzdalenostUzlu >= DIJKSTRA_NEKONECNO ) {
#ifdef DEBUG
            cerr << "preskakuji " << idUzlu << " vzd.: " << vzdalenostUzlu << endl;
#endif // DEBUG
            continue;
        }

        // pro vsechny sousedy, tedy for cyklus pres matici vzdalenosti
        for ( unsigned idSouseda = 0; idSouseda < pocetUzlu; idSouseda++ ) {
            // pokud je uzavreny, preskocim
            if ( uzavreny[idSouseda] == true ) {
                continue;
            }

            vzdalenostSouseda = vzdalenost[idSouseda];
            vzdalenostHrany = graf[idUzlu][idSouseda];
            // kontrola, aby nepretekla hodnota
            if ( vzdalenostHrany >= DIJKSTRA_NEKONECNO )
                novaVzdalenost = DIJKSTRA_NEKONECNO;
            else
                novaVzdalenost = vzdalenostUzlu + vzdalenostHrany;

            // nalezeni kratsi vzdalenosti
            if ( novaVzdalenost < vzdalenostSouseda ) {
#ifdef DEBUG
                cerr << "   nova vzdalenost z " << idUzlu << " do " << idSouseda << " = " << vzdalenostHrany << '(' << novaVzdalenost << ')' << endl;
#endif // DEBUG
                predchudce[idSouseda] = idUzlu;
                vzdalenost[idSouseda] = novaVzdalenost;

#ifdef DEBUG
                vypisFrontu( );
#endif // DEBUG
            }
        }
    }

    return true;
}

bool cDijkstra::inicializace(unsigned idVychozihoUzlu) {
    if ( idVychozihoUzlu >= pocetUzlu ) {
        cerr << "inicializace(): Chyba! id uzlu je vyssi nez pocet uzlu.";
        return false;
    }

    pocetUzavrenychUzlu = 0;
    for ( unsigned idUzlu = 0; idUzlu < pocetUzlu; idUzlu++ ) {
        vzdalenost[idUzlu] = DIJKSTRA_NEKONECNO;
        predchudce[idUzlu] = DIJKSTRA_NEDEFINOVANO;
        uzavreny[idUzlu] = false;
    }
    vzdalenost[idVychozihoUzlu] = 0;

    return true;
}

bool cDijkstra::vyberMinimumZFronty( unsigned & idMinima ) const {
    if (pocetUzavrenychUzlu == pocetUzlu) {
        cerr << "vyberMinimumZFronty(): Chyba! Prazdna fronta!" << endl;
        return false;
    }

    unsigned minimum = DIJKSTRA_NEKONECNO;
    // pro kontrolu, zda se nejaky uzel najde
    idMinima = pocetUzlu;
    // ze vsech neuzavrenych uzlu vybere minimum
    for ( unsigned idUzlu = 0 ; idUzlu < pocetUzlu ; idUzlu++ ) {
       if ( uzavreny[idUzlu] == true ) 
          continue;
       if ( vzdalenost[idUzlu] <= minimum ) {
          minimum  = vzdalenost[idUzlu];
          idMinima = idUzlu;
       }
    }
    // pro kontrolu, zda se nejaky uzel nasel
    if ( idMinima == pocetUzlu ) {
        cerr << "vyberMinimumZFronty(): Neocekavana chyba! Fronta neni prazdna, minimum se ale nenalezlo!" << endl;
        return false;
    }

    return true;
}

void cDijkstra::vypisFrontu( ) const {
   cerr << "Fronta:\n";
   for ( unsigned idUzlu = 0 ; idUzlu < pocetUzlu ; idUzlu++ ) {
      if      ( uzavreny[idUzlu] ) 
         cerr << " z ";
      else if ( vzdalenost[idUzlu] == DIJKSTRA_NEKONECNO ) 
         cerr << " - ";
      else
         cerr << setw(2) << vzdalenost[idUzlu] << " ";
   }
   cerr << endl;
}

void cDijkstra::vypisVysledekPoUzlech(unsigned uzelId) const {
    unsigned hodnota;
    cout << "id uzlu:         ";
    for (unsigned i = 0; i < pocetUzlu; i++) {
        cout << setw(2) << i << " ";
    }
    cout << "\n"
            "Vzdalenosti[" << uzelId << "]:  ";
    for (unsigned i = 0; i < pocetUzlu; i++) {
        hodnota = vzdalenost[i];
        if (hodnota == DIJKSTRA_NEKONECNO)
            cout << " - ";
        else
            cout << setw(2) << hodnota << " ";
    }
    cout << "\n"
            "Predchudci[" << uzelId << "]:   ";
    for (unsigned i = 0; i < pocetUzlu; i++) {
        hodnota = predchudce[i];
        if (hodnota == DIJKSTRA_NEDEFINOVANO)
            cout << " - ";
        else
            cout << setw(2) << hodnota << " ";
    }
    cout << endl;
}

void cDijkstra::vypisVysledekMaticove( ) const {
    cout << "Vzdalenosti:" << endl;
    vypisGrafu(cout, vzdalenostM, pocetUzlu);
    cout << endl << "Predchudci:" << endl;
    vypisGrafu(cout, predchudceM, pocetUzlu);
}

