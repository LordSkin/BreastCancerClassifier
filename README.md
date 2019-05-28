# BreastCancerClassifier

1. dane dla sieci zciągnięto z pliku <B>wdbc.data</B>. W pliku poszczególne wiersze reprezentuja kolejne przypadki uczące.
 
 2. Pierwsza kolumna oznacza id przypadku i nie jest nigdzie potrzebna. Druga kolumna to etykieta - M oraz B czyli przypadek złośliwy oraz nie złośliwy. Te litery zostały zamienione na 0 i 1. Kolejne kolumny to cechy - w sumie jest ich 30.
 
 3. W pliku networks.py znajdują się funkcje wykonujące wszystkie potrzebne operacje do przetestowania sieci,
 a w temp.py przykład ich użycia.