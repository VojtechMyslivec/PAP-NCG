/** cFloydWarshall.cpp
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-brezen 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 sekvencni cast
 *                 trida cFloydWarshall pro algoritmus Floyd-Warshall
 *
 *
 */

#include "cFloydWarshall.h"
#include <iomanip>

cFloydWarshall::cFloydWarshall(unsigned ** graf, unsigned pocetUzlu) {
    this->pocetUzlu = pocetUzlu;
    delkaPredchozi = new unsigned*[pocetUzlu];
    delkaAktualni = new unsigned*[pocetUzlu];
    predchudcePredchozi = new unsigned*[pocetUzlu];
    predchudceAktualni = new unsigned*[pocetUzlu];

    for (unsigned i = 0; i < pocetUzlu; i++) {
        delkaPredchozi[i] = new unsigned[pocetUzlu];
        delkaAktualni[i] = new unsigned[pocetUzlu];
        predchudcePredchozi[i] = new unsigned[pocetUzlu];
        predchudceAktualni[i] = new unsigned[pocetUzlu];
        for (unsigned j = 0; j < pocetUzlu; j++) {
            delkaPredchozi[i][j] = graf[i][j];
            if (i == j || graf[i][j] == FW_NEKONECNO)
                predchudcePredchozi[i][j] = FW_NEDEFINOVANO;
            else
                predchudcePredchozi[i][j] = i;
        }
    }
}

cFloydWarshall::~cFloydWarshall() {
    for (unsigned i = 0; i < pocetUzlu; i++) {
        delete [] delkaPredchozi[i];
        delete [] delkaAktualni[i];
        delete [] predchudcePredchozi[i];
        delete [] predchudceAktualni[i];
    }
    delete [] delkaPredchozi;
    delete [] delkaAktualni;
    delete [] predchudcePredchozi;
    delete [] predchudceAktualni;
}

void cFloydWarshall::spustVypocet() {
    unsigned novaVzdalenost;

    for (unsigned k = 0; k < pocetUzlu; k++) {
        for (unsigned i = 0; i < pocetUzlu; i++) {
            for (unsigned j = 0; j < pocetUzlu; j++) {
                // osetreni nekonecna
                if (delkaPredchozi[i][k] == FW_NEKONECNO || delkaPredchozi[k][j] == FW_NEKONECNO)
                    novaVzdalenost = FW_NEKONECNO;
                else
                    novaVzdalenost = delkaPredchozi[i][k] + delkaPredchozi[k][j];

                // pokud nalezne kratsi cestu, zapise ji a zmeni predchudcePredchozi
                if (novaVzdalenost < delkaPredchozi[i][j]) {
                    delkaAktualni[i][j] = novaVzdalenost;
                    predchudceAktualni[i][j] = predchudcePredchozi[k][j];
                }// jinak delka i predchudce zustavaji
                else {
                    delkaAktualni[i][j] = delkaPredchozi[i][j];
                    predchudceAktualni[i][j] = predchudcePredchozi[i][j];
                }
            }
        }

        // prohozeni predchozi a aktualni
        prohodPredchoziAAktualni();
    }

    // prohozeni predchozi a aktualni, aby vysledky byly v aktualnim (po skonceni cyklu jsou vysledky v predchozim)
    prohodPredchoziAAktualni();
}

void cFloydWarshall::prohodPredchoziAAktualni() {
    unsigned ** pomocny;
    
    pomocny = delkaPredchozi;
    delkaPredchozi = delkaAktualni;
    delkaAktualni = pomocny;

    pomocny = predchudcePredchozi;
    predchudcePredchozi = predchudceAktualni;
    predchudceAktualni = pomocny;
}


void cFloydWarshall::vypisVysledekMaticove() const {
    vypisGrafu(cout, delkaAktualni, pocetUzlu);
    cout << endl;
    vypisGrafu(cout, predchudceAktualni, pocetUzlu);
}

void cFloydWarshall::vypisVysledekPoUzlech() const {
    unsigned hodnota;

    for (unsigned i = 0; i < pocetUzlu; i++) {
        cout << "\nFloyd-Warshall pro uzel id = " << i << endl;
        cout << "id uzlu:         ";
        for (unsigned j = 0; j < pocetUzlu; j++) {
            cout << setw(2) << j << " ";
        }
        cout << "\n"
                "Vzdalenosti[" << i << "]:  ";
        for (unsigned j = 0; j < pocetUzlu; j++) {
            hodnota = delkaAktualni[i][j];
            if (hodnota == FW_NEKONECNO)
                cout << " - ";
            else
                cout << setw(2) << hodnota << " ";
        }

        cout << "\n"
                "Predchudci[" << i << "]:   ";
        for (unsigned j = 0; j < pocetUzlu; j++) {
            hodnota = predchudceAktualni[i][j];
            if (hodnota == FW_NEDEFINOVANO)
                cout << " - ";
            else
                cout << setw(2) << hodnota << " ";
        }
        cout << endl;
    }
}

