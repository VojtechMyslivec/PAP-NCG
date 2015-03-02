/* 
 * File:   cFloyd_Warshall.cpp
 * Author: novy
 * 
 * Created on 2. březen 2015, 7:15
 */

#include "cFloyd_Warshall.h"

cFloyd_Warshall::cFloyd_Warshall(unsigned ** graf, unsigned pocetUzlu) {
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

cFloyd_Warshall::cFloyd_Warshall(const cFloyd_Warshall& orig) {
}

cFloyd_Warshall::~cFloyd_Warshall() {
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

void cFloyd_Warshall::spustVypocet() {
    unsigned ** pomocny;
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
        pomocny = delkaPredchozi;
        delkaPredchozi = delkaAktualni;
        delkaAktualni = pomocny;

        pomocny = predchudcePredchozi;
        predchudcePredchozi = predchudceAktualni;
        predchudceAktualni = pomocny;
    }
}

void cFloyd_Warshall::vypisVysledekMaticove() const {
    vypisGrafu(cout, delkaAktualni, pocetUzlu);
    cout << endl;
    vypisGrafu(cout, predchudceAktualni, pocetUzlu);
}

void cFloyd_Warshall::vypisVysledekPoUzlech() const {
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
            "Predchudci["<<i<<"]:   ";
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


