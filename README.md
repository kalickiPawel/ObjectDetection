# Projekt: "detekcja obiektów" (Python)

- Z wykorzystaniem biblioteki OpenCV (w Pythonie pakiet cv2) przygotuj funkcje do: wczytywania obrazów, przeskalowania ich (z ustaleniem docelowej wysokości 480), konwersji kolorów do szarości.
- Przygotuj zestaw funkcji do obliczania cech Haara przy zadanej parametryzacji (liczba skal, rozmiar siatki punktów zaczepienia). Potrzebne tu będzie określenie szablonów cech Haara oraz funkcje m.in. do: wyznaczenia obrazu całkowego, obliczenia przyrostu obrazu całkowego nad zadanym prostokątem, wyznaczenia współrzędnych cech Haara (znormalizowanych do kwadratu jednostkowego), wyznaczenia pojedynczej cechy Haara (gdy znane są już jej współrzędne piskelowe), wyznaczenia wektora wszystkich cech Haara (dla danej parametryzacji).
- Zwizualizuj cechy i wypisz ich wartości na przykładowym obrazie.
- Zgodnie ze wskazówkami prowadzącego użyj bazy FDDB (Face Detection Database and Benchmark [1]) w celu wygenerowania zbiorów danych (tablice numpy) z cechami Haara do uczenia i testowania detektorów.
- Wykorzystując bibliotekę scikit-learn naucz klasyfikator AdaBoost + decision stumps jako detektor twarzy (sugerowana do pierwszych eksperymentów liczba rund: 128).
- Zmierz dokładność uczącą i testową otrzymanego klasyfikatora.
- Zwizualizuj najważniejsze cechy wybrane przez algorytm uczący.
- Napisz funkcję realizującą procedurę detekcyjną skanującą obraz oknem przesuwnym (i z wykorzystaniem nauczonego klasyfikatora). Sugerowane nastawy początkowe: okno minimalne - 64 x 64, liczba skal - 4, współczynnik wzrostu rozmiaru okna - 1.2, współczynnik skoków okna - 0.1. W ramach procedury detekcyjnej wykonaj wstępne zliczenie liczby okien, które będą badane. Uwaga: na rzecz każdego okna ekstrahuj tylko podzbiór wybranych cech Haara.
- Spróbuj przyspieszyć detekcję poprzez zrównoleglenie (pakiet multiprocessing) oraz niskopoziomowe skompilowanie wybranych funkcji (pakiet numba).

## Zadanie domowe

- Zaprogramuj własny algorytm uczący RealBoost + bins (zgodnie z konwencją scikit-learn). Zadbaj o szybkość uczenia i zwracania odpowiedzi (unikanie pętli, obliczenia na macierzach numpy).
- Naucz detektor własnym algorytmem i podepnij do procedury detekcyjnej.
- Opracuj funkcję do grupowania klastrów wykrytych okien (odpowiednio mocno nakładających się, wykorzystaj miarę IoU). Funkcja ta powinna być wywołana w celu końcowego przetworzenia bezpośrednich wyników procedury detekcyjnej.
- Wygeneruj krzywą ROC detektora dla zbioru testowego. Wykreśl ją (oś FAR w skali logarytmicznej) i wybierz próg decyzyjny odpowiadający największej dokładności testowej.
- Wykonaj wsadowo funkcję detekcyjną na rzecz wszystkich obrazów z paczki testowej (z użyciem wybranego progu detekcyjnego). Na podstawie informacji o oczekiwanym położeniu obiektów do wykrycia (ground truth) wyznacz dokładność detektora (w szczególności raportuj: czułość - odsetek prawdziwych pozytywów oraz FAR - odesetek fałszywych alarmów przeciętnie na 1 obraz). Ponownie wykorzystaj miarę IoU (z progiem 0.5), aby zdecydować czy wskazanie detektora pokrywa się odpowiednio mocno z oczekiwanym oknem (ground truth). Przygotuj możliwość wizualizacji (lub zapisu obrazów z zaznaczonymi detekcjami) tego wsadowego testu.

## Dane

- FDDB-folds.tar 

http://vis-www.cs.umass.edu/fddb/index.html#download

Vidit Jain and Erik Learned-Miller.
FDDB: A Benchmark for Face Detection in Unconstrained Settings.
Technical Report UM-CS-2010-009, Dept. of Computer Science, University of Massachusetts, Amherst. 2010. 
- originalPics.tar -> 