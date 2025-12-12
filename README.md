# Detekce klatrinem potažených jamek

## Abstrakt
Tento projekt se zabývá aplikací metod strojového učení při analýze mikroskopických dat TIRF-SIM s cílem detekovat a sledovat klatrinem potažené jamky (CCP) během endocytózy. Výchozí baseline metody (jednoduchý morfologický detektor, CNN a Nearest-Neighbour tracker) vykazovaly nízkou přesnost, a proto byly implementovány a experimentálně porovnány pokročilejší přístupy.

V projektu byly porovnávány různé architektury neuronových sítí (U-Net, Res-U-Net), různé postupy filtrování výstupu a několik metod trackingu ( Kalmanův filtr, Nearest-Neighbour přístup a knihovna Norfair). Součástí práce bylo rovněž ladění parametrů (threshold, max_dist, sigma).

Přestože se nepodařilo překonat referenční metodu SOTA, dosažené výsledky výrazně překonaly původní baseline a přiblížily se kvalitě SOTA v metrikách HOTA, AssA a DetA. Projekt tak demonstruje, jak lze kombinací pokročilejších segmentačních modelů a sofistikovanějšího trackingu významně zlepšit analýzu dynamiky CCP.

## Soubory
- Hlavním souborem je SU2_main
- Ve složce [src](https://github.com/Strojove-uceni/su2-final-projects-2025-nesnerova/tree/main/src) jsou skripty obsahující definice modelů
- Soubory  [training_unet](https://raw.githubusercontent.com/Strojove-uceni/su2-final-projects-2025-nesnerova/refs/heads/main/training_unet.ipynb)  a [training_resunet](https://raw.githubusercontent.com/Strojove-uceni/su2-final-projects-2025-nesnerova/refs/heads/main/training_resunet.ipynb) představují trénování dvou modelů, přičemž data z trénování jsou do hlavního souboru stažena z Google Drive
  
