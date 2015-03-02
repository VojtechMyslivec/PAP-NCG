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
 *                 trida cDijkstra pro Dijkstruc algoritmus
 *
 *
 */

#include "cDijkstra.h"

#include <iostream>
#include <iomanip>

using namespace std;
// ============================================================================

cDijkstra::cDijkstra(unsigned ** graf, unsigned pocetUzlu) {
    // TODO melka kopie
    this->graf = graf;
    this->pocetUzlu = pocetUzlu;

    halda = new unsigned[pocetUzlu];
    indexyVHalde = new unsigned[pocetUzlu];
    vzdalenost = new unsigned[pocetUzlu];
    vzdalenostM = new unsigned*[pocetUzlu];
    predchudce = new unsigned[pocetUzlu];
    predchudceM = new unsigned*[pocetUzlu];
    uzavreny = new bool[pocetUzlu];
    velikostHaldy = 0;

    idInstance = 0;
    for (unsigned i = 0; i < pocetUzlu; i++) {
        vzdalenostM[i] = new unsigned[pocetUzlu];
        predchudceM[i] = new unsigned[pocetUzlu];
    }
}

cDijkstra::~cDijkstra() {
    if (halda != NULL)
        delete [] halda;
    if (indexyVHalde != NULL)
        delete [] indexyVHalde;
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

bool cDijkstra::spustVypocet(unsigned idVychozihoUzlu) {
    unsigned idUzlu;
    unsigned vzdalenostUzlu, vzdalenostSouseda, vzdalenostHrany, novaVzdalenost;
    idInstance = idVychozihoUzlu;
    
    // inicializace haldy, vzalenosti, predchudcu
    if (inicializace(idVychozihoUzlu) != true) {
        return false;
    }


#ifdef DEBUG
    vypisHaldu();
#endif // DEBUG

    // dokud nenavstivi vsechny uzly
    while (jePrazdnaHalda() != true) {
        // uzly navstevuje podle nejmensi vzdalenosti
        vemMinimumZHaldy(idUzlu);
        uzavreny[idUzlu] = true;
#ifdef DEBUG
        cerr << "\nzpracovavam uzel " << idUzlu << endl;
        vypisHaldu();
#endif // DEBUG
        vzdalenostUzlu = vzdalenost[idUzlu];
        // pokud je stale nekonecna, znamena to nedostupny uzel;
        if (vzdalenostUzlu >= DIJKSTRA_NEKONECNO) {
#ifdef DEBUG
            cerr << "preskakuji " << idUzlu << " vzd.: " << vzdalenostUzlu << endl;
#endif // DEBUG
            continue;
        }

        // pro vsechny sousedy, tedy for cyklus pres matici vzdalenosti
        for (unsigned idSouseda = 0; idSouseda < pocetUzlu; idSouseda++) {
            // pokud je uzavreny, preskocim
            if (uzavreny[idSouseda] == true) {
                continue;
            }

            vzdalenostSouseda = vzdalenost[idSouseda];
            vzdalenostHrany = graf[idUzlu][idSouseda];
            // kontrola, aby nepretekla hodnota
            if (vzdalenostHrany >= DIJKSTRA_NEKONECNO)
                novaVzdalenost = DIJKSTRA_NEKONECNO;
            else
                novaVzdalenost = vzdalenostUzlu + vzdalenostHrany;

            // nalezeni kratsi vzdalenosti
            if (novaVzdalenost < vzdalenostSouseda) {
#ifdef DEBUG
                cerr << "   nova vzdalenost z " << idUzlu << " do " << idSouseda << " = " << vzdalenostHrany << '(' << novaVzdalenost << ')' << endl;
#endif // DEBUG
                predchudce[idSouseda] = idUzlu;
                predchudceM[idInstance][idSouseda] = idUzlu;
                nastavVzdalenostUzlu(idSouseda, novaVzdalenost);

                // oprava haldy dle nove hodnoty
                opravPoziciVHalde(indexyVHalde[idSouseda]);
#ifdef DEBUG
                vypisHaldu();
#endif // DEBUG
            }
        }
    }

    return true;
}

bool cDijkstra::inicializace(unsigned idVychozihoUzlu) {
    if (idVychozihoUzlu >= pocetUzlu) {
        cerr << "inicializace(): Chyba! id uzlu je vyssi nez pocet uzlu.";
        return false;
    }

    velikostHaldy = 0;
    for (unsigned idUzlu = 0; idUzlu < pocetUzlu; idUzlu++) {
        vzdalenost[idUzlu] = DIJKSTRA_NEKONECNO;
        vzdalenostM[idVychozihoUzlu][idUzlu] = DIJKSTRA_NEKONECNO;
        predchudce[idUzlu] = DIJKSTRA_NEDEFINOVANO;
        predchudceM[idVychozihoUzlu][idUzlu] = DIJKSTRA_NEDEFINOVANO;
        uzavreny[idUzlu] = false;
        pridejPrvekDoHaldy(idUzlu);
    }
    // zde neni kontrola navratove hodnoty -- zbytecne, uz je zkontrolovana
    nastavVzdalenostUzlu(idVychozihoUzlu, 0);

    return true;
}

void cDijkstra::nastavVzdalenostUzlu(unsigned idUzlu, unsigned novaVzdalenost) {
    vzdalenost[idUzlu] = novaVzdalenost;
    vzdalenostM[idInstance][idUzlu] = novaVzdalenost;
    opravPoziciVHalde(indexyVHalde[idUzlu]);
}

void cDijkstra::opravPoziciVHalde(unsigned pozice) {
    // nalezne misto, kam ma uzel na pozici patrit
    // indexDoHaldy ukazuje na opravovane misto, indexOtceVHalde na otce tohoto mista
    // idUzluOtce je hodnota v halde na pozici otce a hodnotaUzlu je vzdalenost uzlu na pozici
    unsigned idUzlu = halda[pozice];
    unsigned hodnotaUzlu = vzdalenost[idUzlu];
#ifdef DEBUG2
    cerr << " + oprav pozici " << pozice << "; id uzlu " << idUzlu << "; hodnota Uzlu " << hodnotaUzlu << endl;
#endif // DEBUG2

    unsigned indexDoHaldy, indexOtceVHalde, idUzluOtce;
    for (indexDoHaldy = pozice; indexDoHaldy > 0; indexDoHaldy = otec(indexDoHaldy)) {
        indexOtceVHalde = otec(indexDoHaldy);
        idUzluOtce = halda[indexOtceVHalde];
#ifdef DEBUG2
        cerr << " ++ indexOtceVHalde " << indexOtceVHalde << "; idUzluOtce " << idUzluOtce << "; vzdalenost otce ";
        if (vzdalenost[idUzluOtce] == DIJKSTRA_NEKONECNO)
            cerr << " - ";
        else
            cerr << vzdalenost[idUzluOtce];
        cerr << endl;
#endif // DEBUG2

        // pokud je vzdalenost otce mensi nez hodnota uzlu, skonci a indexDoHaldy ukazuje 
        // na spravnou pozici
        if (vzdalenost[idUzluOtce] <= hodnotaUzlu)
            break;

        // presun otce indexu na nizsi pozici
        halda[indexDoHaldy] = idUzluOtce;
        indexyVHalde[idUzluOtce] = indexDoHaldy;
    }

    // na nalezene (uvolnene) misto ulozi id puvodniho uzlu
    halda[indexDoHaldy] = idUzlu;
    indexyVHalde[idUzlu] = indexDoHaldy;

}

bool cDijkstra::pridejPrvekDoHaldy(unsigned idUzlu) {
    if (velikostHaldy >= pocetUzlu) {
        cerr << "vemMinimumZHaldy(): Chyba! Preteceni haldy!" << endl;
        return false;
    }

    // prida idUzlu na prvni volne misto v halde, opravi haldu a zvysi pocet prvku v halde
    halda[velikostHaldy] = idUzlu;
    indexyVHalde[idUzlu] = velikostHaldy;
    opravPoziciVHalde(velikostHaldy);
    velikostHaldy++;

    return true;
}

void cDijkstra::haldujRekurzivne(unsigned indexOtce) {
    unsigned indexLeveho = levy(indexOtce);
    unsigned indexPraveho = pravy(indexOtce);

    unsigned vzdalenostOtce = vzdalenost[halda[indexOtce]];
#ifdef DEBUG2
    cerr << " +       indexOtce " << indexOtce << ";      indexLeveho " << indexLeveho << ";      indexPraveho " << indexPraveho << endl;
#endif //DEBUG2
    // nejmensi ze tri
    unsigned indexNejmensiho, vzdalenostNejmensiho, pomocny;
    if (indexLeveho < velikostHaldy && vzdalenost[halda[indexLeveho]] < vzdalenostOtce)
        indexNejmensiho = indexLeveho;
    else
        indexNejmensiho = indexOtce;
    vzdalenostNejmensiho = vzdalenost[halda[indexNejmensiho]];

    if (indexPraveho < velikostHaldy && vzdalenost[halda[indexPraveho]] < vzdalenostNejmensiho)
        indexNejmensiho = indexPraveho;
    vzdalenostNejmensiho = vzdalenost[halda[indexNejmensiho]];

#ifdef DEBUG2
    cerr << " ++ vzdalenostOtce ";
    if (vzdalenostOtce == DIJKSTRA_NEKONECNO)
        cerr << '-';
    else
        cerr << vzdalenostOtce;
    cerr << "; indexNejmensiho " << indexNejmensiho << "; vzdalenostNejmensiho " << vzdalenostNejmensiho << endl;
#endif //DEBUG2

    // pokud je pod otcem mensi prvek, prohod je a zavolej se rekurzivne
    if (indexNejmensiho != indexOtce) {
        pomocny = halda[indexOtce];
        halda[indexOtce] = halda[indexNejmensiho];
        halda[indexNejmensiho] = pomocny;

        indexyVHalde[halda[indexOtce]] = indexOtce;
        indexyVHalde[halda[indexNejmensiho]] = indexNejmensiho;

        // prvek byl presunut na pozici nejmensi (pravy ci levy syn), halduj tam
        haldujRekurzivne(indexNejmensiho);
    }
    // jinak je otec na spravnem miste
}

bool cDijkstra::vemMinimumZHaldy(unsigned & idUzlu) {
    if (velikostHaldy == 0) {
        cerr << "vemMinimumZHaldy(): Chyba! Podteceni haldy!" << endl;
        return false;
    }
    // minimum je na vrchu haldy
    idUzlu = halda[0];
    indexyVHalde[idUzlu] = DIJKSTRA_NEDEFINOVANO;
    // posledni prvek ulozi na vrchol, zmensi pocet prvku a opravi haldu
    velikostHaldy--;
    halda[0] = halda[velikostHaldy];
    indexyVHalde[halda[0]] = 0;

    haldujRekurzivne(0);

    return true;
}

bool cDijkstra::jePrazdnaHalda() const {
    return velikostHaldy == 0;
}

void cDijkstra::vypisHaldu() const {
    cerr << "Halda: \n";
    unsigned pravy = 0;
    for (unsigned i = 0; i < velikostHaldy; i++) {
        if (i > pravy) {
            cerr << '\n';
            pravy = pravy(pravy);
        }
        cerr << halda[i] << '(';
        if (vzdalenost[halda[i]] == DIJKSTRA_NEKONECNO)
            cerr << '-';
        else
            cerr << vzdalenost[halda[i]];
        cerr << ") ";
        if (i % 2 == 0)
            cerr << ' ';
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

