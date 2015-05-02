/** funkceSpolecne.cu
 *
 * Autori:      Vojtech Myslivec <vojtech.myslivec@fit.cvut.cz>,  FIT CVUT v Praze
 *              Zdenek  Novy     <novyzde3@fit.cvut.cz>,          FIT CVUT v Praze
 *              
 * Datum:       unor-duben 2015
 *
 * Popis:       Semestralni prace z predmetu MI-PAP:
 *              Hledani nejkratsich cest v grafu 
 *                 paralelni implementace na CUDA
 *                 Spolecne funkce pro nacitani / vypis dat
 *
 *
 */

#include "funkceSpolecne.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unistd.h> // getopts()

using namespace std;

void HandleError( cudaError_t chyba, const char * soubor, int radek ) {
    if ( chyba != cudaSuccess ) {
        fprintf( stderr, "%s v %s na radku %d\n", cudaGetErrorString( chyba ), soubor, radek );
        exit( 1 );
    }
}


void vypisUsage( ostream & os, const char * jmenoProgramu ) {
    os << "USAGE\n"
          "   " << jmenoProgramu << " [-w pocet_warpu] -f vstupni_soubor\n"
          "   " << jmenoProgramu << " -h\n"
          "\n"
          "      vstupni_soubor   Soubor s daty grafu ve formatu ohodnocene incidencni\n"
          "                       matice (hrany ohodnocene vahami).\n"
          "                       Jedna se o n^2 + 1 unsigned hodnot, kde prvni hodnota\n"
          "                       udava cislo n (pocet uzlu, ruzny od nuly). Hodnota -\n"
          "                       urcuje neexistujici hranu\n"
          "\n"
          "      pocet_warpu      Pocet warpu v jednom bloku pro paralelni cast vypoctu.\n"
          "                       Pocet bloku se dopocita automaticky, aby pocet vlaken\n"
          "                       byl >= pocet uzlu grafu.\n"
          "                       Vychozi hodnota: 4\n"
          "\n"
          "      -h               Vypise tuto napovedu a skonci."
        << endl;
}

void uklid( unsigned ** graf, unsigned pocetUzlu ) {
    if ( graf != NULL ) {
        for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
            if ( graf[i] != NULL ) {
                // page-locked memory
                HANDLE_ERROR( 
                        cudaFreeHost( graf[i] )
                        );
                graf[i] = NULL;
            }
        }
        // page-locked memory
        HANDLE_ERROR( 
                cudaFreeHost( graf )
                );
        graf = NULL;
    }
}

bool zkontrolujPrazdnyVstup( istream & is ) {
    char c = '\0';
    while ( true ) {
        c = is.get();
        if ( is.fail() ) 
            break;
        if ( ! ( c == ' ' || c == '\t' || c == '\n' || c == '\r' ) )
            return false;
    }
    return true;
}

bool nactiPocetWarpu( const char * vstup, unsigned & pocetWarpu ) {
    istringstream iss( vstup );
    int tmp;
    iss >> tmp;
    if ( tmp < 1         ||
            iss.fail( )     ||
            zkontrolujPrazdnyVstup( iss ) != true 
       ) {
        return false;
    }
    if ( tmp > CUDA_MAX_POCET_WARPU ) {
        return false;
    }
    pocetWarpu = (unsigned)tmp;
    return true;
}

bool zkontrolujSoubor( const char * optarg ) {
    ifstream f( optarg );
    bool navrat = true;
    if ( f.fail() ) 
        navrat = false;
    f.close();
    return navrat;
}

bool parsujArgumenty( int argc, char ** argv, char *& souborSGrafem, unsigned & pocetWarpu, unsigned & navrat ) {
    if ( argc < 2 ) {
        cerr << "Nedostatecny pocet argumentu" << endl;
        vypisUsage( cerr, argv[0] );
        navrat = MAIN_ERR_USAGE;
        return false;
    }

    int o = 0;
    souborSGrafem = NULL;
    opterr = 0; // zabrani getopt, aby vypisovala chyby
    // optstring '+' zajisti POSIX zadavani prepinacu, 
    //           ':' pro rozliseni neznameho prepinace od chybejiciho argumentu
    while ( ( o = getopt( argc, argv, "+:hf:w:" ) ) != -1 ) {
        switch ( o ) {
            case 'h':
                vypisUsage( cout, argv[0] );
                navrat = MAIN_OK;
                return false;

            case 'f':
                if ( zkontrolujSoubor( optarg ) != true ) {
                    cerr << argv[0] << ": Soubor s daty neexistuje nebo neni citelny!" << endl;
                    navrat = MAIN_ERR_VSTUP;
                    return false;
                }
                souborSGrafem = optarg;
                break;

            case 'w':
                if ( nactiPocetWarpu( optarg, pocetWarpu ) != true ) {
                    cerr << argv[0] << ": Pocet warpu musi byt kladne cele cislo mensi rovno nez " << CUDA_MAX_POCET_WARPU << endl;
                    navrat = MAIN_ERR_VSTUP;
                    return false;
                }
                break;

            case ':':
                cerr << argv[0] << ": Prepinac '-" << (char)optopt << "' vyzaduje argument." << endl;
                navrat = MAIN_ERR_VSTUP;
                return false;

            case '?':
                cerr << argv[0] << ": Neznamy prepinac '-" << (char)optopt << "'." << endl;
                navrat = MAIN_ERR_VSTUP;
                return false;

            default:
                navrat = MAIN_ERR_NEOCEKAVANA;
                return false;
        }
    }
    if ( souborSGrafem == NULL ) {
        cerr << argv[0] << ": Musi byt urceno jmeno souboru s grafem (prepinac '-f')" << endl;
        navrat = MAIN_ERR_VSTUP;
        return false;
    }
    if ( optind != argc ) {
        cerr << argv[0] << ": Chyba pri zpracovani parametru, nejsou dovoleny zadne argumenty." << endl;
        navrat = MAIN_ERR_VSTUP;
        return false;
    }

    return true;
}

unsigned nactiHodnotu( istream & is, unsigned & hodnota ) {
    char c;
    // sebrani prazdnych znaku
    do {
        is.get( c );
        // pokud je prazdny vstup, skonci
        if ( is.fail() ) {
            return NACTI_ERR_PRAZDNO;
        }
    } while ( c == ' ' || c == '\t' || c == '\n' );

    // na vstupu je - 
    if ( c == '-' ) {
        // zkontroluje, jestli nasleduje prazdny znak   
        is.get( c );
        if ( c != ' ' && c != '\t' && c != '\n' ) {
            // pokud nasledovalo cislo, byl to pokus o zadani zaporne hodnoty
            if ( c >= '0' && c <= '9') {
                return NACTI_ERR_ZNAMENKO;
            }
            // jinak jde o nejaky spatny vstup (text)
            return NACTI_ERR_TEXT;
        }

        // kolem znaku - jsou spravne prazdne znaky, reprezentuje tedy nekonecno
        hodnota = NEKONECNO;
        return NACTI_NEKONECNO;
    }
    // jinak to ma byt cislo

    // vrati znak na vstup a pokusi se nacist cislo
    is.putback( c );
    is >> hodnota;
    // pokud se nepodarilo nacist, nebylo na vstupu cislo
    if ( is.fail( ) ) {
        return NACTI_ERR_CISLO;
    }
    return NACTI_OK;
}

bool nactiGraf( istream & is, unsigned ** & graf, unsigned & pocetUzlu ) {
    is >> pocetUzlu;
    if ( is.fail( ) || pocetUzlu < 1 ) {
        cerr << "nactiGraf(): Chyba! Pocet uzlu musi byt kladne cislo" << endl;
        return false;
    }

    // page-locked memory
    HANDLE_ERROR( 
            cudaHostAlloc( 
                &graf, 
                pocetUzlu * sizeof(*graf),
                cudaHostAllocDefault
                )
            );
    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
        // page-locked memory
        HANDLE_ERROR( 
                cudaHostAlloc( 
                    &graf[i], 
                    pocetUzlu * sizeof(*graf[i]),
                    cudaHostAllocDefault
                    )
                );
    }

    unsigned ret;
    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
        for ( unsigned j = 0 ; j < pocetUzlu ; j++ ) {
            ret = nactiHodnotu( is, graf[i][j] );
            // pouze se rozhoduje vypis pripadne chybove hlasky
            switch ( ret ) {
                case NACTI_OK:
                case NACTI_NEKONECNO:
                    continue;
                case NACTI_ERR_PRAZDNO:
                    cerr << "nactiGraf(): Chyba! V ocekavam n^2 nezapornych hodnot (n = " << pocetUzlu << ")." << endl;
                    return false;
                case NACTI_ERR_ZNAMENKO:
                case NACTI_ERR_CISLO:
                case NACTI_ERR_TEXT:
                    cerr << "nactiGraf(): Chyba! Ocekavam pouze nezaporne hodnoty nebo '-' pro nekonecno." << endl;
                    return false;
                default:
                    cerr << "nactiGraf(): Neocekavana chyba vstupu." << endl;
                    return false;
            }
        }
    }

    if ( zkontrolujPrazdnyVstup( is ) != true ) {
        cerr << "nactiGraf(): Chyba! Na vstupu je vice dat, nez je ocekavano (n = " << pocetUzlu << ")." << endl;
        return false;
    }
    return true;
}

bool nactiData( const char * jmenoSouboru, unsigned ** & graf, unsigned & pocetUzlu ) {
    ifstream ifs( jmenoSouboru );
    if ( ifs.fail( ) ) {
        cerr << "nactiData(): Chyba! Nelze otevrit soubor '" << jmenoSouboru << "' pro cteni." << endl;
        return false;
    }
    bool ret = nactiGraf( ifs, graf, pocetUzlu );
    ifs.close( );
    return ret;
}

unsigned kontrolaGrafu( unsigned ** graf, unsigned pocetUzlu ) {
    unsigned ret = GRAF_NEORIENTOVANY;

    for ( unsigned i = 0 ; i < pocetUzlu - 1; i++ ) {
        if ( graf[i][i] != 0 ) {
            cerr << "Chyba formatu! Hodnota w(i,i) musi byt 0!"
                "w(" << i << ',' << i << ") = " << graf[i][i] << endl;
            return GRAF_CHYBA;
        }

        for ( unsigned j = i + 1; j < pocetUzlu ; j++ ) {
            if ( graf[i][j] != graf[j][i] ) {
                ret = GRAF_ORIENTOVANY;
            }
        }
    }
    return ret;
}

void vypisGrafu( ostream & os, unsigned ** graf, unsigned pocetUzlu ) {
    os << setw(2) << ' ' << " |";
    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) 
        os << setw(2) << i << ' ';
    os << "\n---";
    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) 
        os << "---";


    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
        os << '\n' << setw(2) << i << " |";
        for ( unsigned j = 0 ; j < pocetUzlu ; j++ ) {
            if ( graf[i][j] == NEKONECNO )
                os << " - ";
            else
                os << setw(2) << graf[i][j] << ' ';
        }
    }
    os << endl;
}



void maticeInicializaceNaGPU( unsigned ** graf, unsigned pocetUzlu, unsigned **& devGraf ) {
    // TODO #optimalizace
    //    alokovat 2D pole pomoci pitch
    // alokace matice -- pole sloupcu ---------------------
    HANDLE_ERROR( 
            cudaMalloc( 
                &devGraf,
                pocetUzlu*sizeof(*devGraf)
                )
            ); 

    // v cyklu se alokuji a kopiruji data z grafu na GPU
    unsigned * devHodnoty;
    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
        // alokace jednoho radku matice -------------------
        HANDLE_ERROR( 
                cudaMalloc( 
                    &devHodnoty,
                    pocetUzlu*sizeof(*devHodnoty)
                    )
                );
        // kopirovani jednoho radku matice ----------------
        if ( graf != NULL ) {
            HANDLE_ERROR( 
                    cudaMemcpy( 
                        devHodnoty,
                        graf[i],
                        pocetUzlu*sizeof(*devHodnoty), 
                        cudaMemcpyHostToDevice 
                        )
                    );
        }
        // zkopirovani pointeru na radek do pole sloupcu --
        HANDLE_ERROR( 
                cudaMemcpy( 
                    &(devGraf[i]),
                    &(devHodnoty),
                    sizeof(devHodnoty),
                    cudaMemcpyHostToDevice
                    )
                );
    }
}

void maticeInicializaceNaCPU( unsigned **& hostMatice, unsigned pocetUzlu ) {
    // page-lock memory
    HANDLE_ERROR( 
            cudaHostAlloc( 
                &hostMatice, 
                pocetUzlu * sizeof(*hostMatice),
                cudaHostAllocDefault
                )
            );
    for ( unsigned i = 0; i < pocetUzlu; i++ ) {
        // page-lock memory
        HANDLE_ERROR( 
                cudaHostAlloc( 
                    &(hostMatice[i]), 
                    pocetUzlu * sizeof(*hostMatice[i]),
                    cudaHostAllocDefault
                    )
                );
        for ( unsigned j = 0; j < pocetUzlu; j++ ) {
            hostMatice[i][j] = NEKONECNO;
        }
    }
}

void maticeUklidNaGPU( unsigned **& devGraf, unsigned pocetUzlu ) {
    // v cyklu se kopiruji hodnoty ukazatelu na radek a ty se smazou
    unsigned * devHodnoty;
    for ( unsigned i = 0 ; i < pocetUzlu ; i++ ) {
        // zkopirovani ukazatele z [pole sloupcu na device] --
        HANDLE_ERROR( 
                cudaMemcpy( 
                    &(devHodnoty),
                    &(devGraf[i]),
                    sizeof(devHodnoty),
                    cudaMemcpyDeviceToHost
                    )
                );
        // uvolneni pameti radku matice -------------------
        HANDLE_ERROR( 
                cudaFree( devHodnoty )
                );
    }
    // uvolneni pameti ukazatele na radky matice ----------
    HANDLE_ERROR( 
            cudaFree( devGraf )
            ); 
    devGraf = NULL;
}

void maticeUklidNaCPU( unsigned **& hostMatice, unsigned pocetUzlu ) {
    if ( hostMatice != NULL ) {
        for ( unsigned i = 0; i < pocetUzlu; i++ ) {
            if ( hostMatice[i] != NULL ) {
                // page-locked memory
                HANDLE_ERROR( 
                        cudaFreeHost( hostMatice[i] )
                        );
                hostMatice[i] = NULL;
            }
        }
        // page-locked memory
        HANDLE_ERROR( 
                cudaFreeHost( hostMatice )
                );
        hostMatice = NULL;
    }
}


#ifdef MERENI
void mereniInicializace( cudaEvent_t udalosti[], unsigned pocet ) {
    for ( unsigned i = 0 ; i < pocet ; i++ ) {
        HANDLE_ERROR(   cudaEventCreate( &(udalosti[i]) )   );
    }
}

void mereniZaznam( cudaEvent_t udalost ) {
    // event synchronize, aby se vsechny operace dokoncily a mereni
    // probehlo v poradku
    HANDLE_ERROR(   cudaEventRecord(      udalost )    );
    HANDLE_ERROR(   cudaEventSynchronize( udalost )    );
}

void mereniUplynulo( float & cas, cudaEvent_t zacatek, cudaEvent_t konec ) {
    HANDLE_ERROR(   cudaEventElapsedTime( &cas, zacatek, konec )   );
    // prepocet na sekundy
    cas = cas / 1000;
}

void mereniUklid( cudaEvent_t udalosti[], unsigned pocet ) {
    for ( unsigned i = 0 ; i < pocet ; i++ ) {
        HANDLE_ERROR(   cudaEventDestroy( udalosti[i] )   );
    }
}

#endif // MERENI

