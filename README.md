# Taio
1. Githubowe akcje wywalą się, jeśli pushowane pliki nie spełniają standardów Pythona.
2. Po ściągnięciu kodu można wywołać `pip install -r requirements.txt`, by mieć biblioteki wykorzystywane w projekcie (są jeszcze lepsze rozwiązania, ale chyba nie warto w tym projekcie).
3. Jak doinstalowujemy sobie jakiś pakiet i ktoś go może nie mieć to fajnie jest to dodać w `requirements.txt`.
4. W pakietach w `requirements.txt` jest też `black` do automatycznego refactoru. Korzystamy z niego jako `black jakis_plik.py`. Nie zawsze jego zmiany są fajne, ale pozwala utrzymać w miarę spójny standard i ogólnie jest polecany.
5. Pakietu `pylint` możemy użyć też lokalnie (to jest ten co ewentualnie wywali akcje na Githubie) - polecam przed pushowaniem. Wywołanie `pylint jakis_plik.py`. Uwzględniony jest w `requirements.txt`.
**UWAGA!** Lokalnie jak dostajecie warningi o zmiennych jednoliterowych to się tym nie przejmujcie, github to przepuści. Możecie ewentualnie lokalnie uruchomić z opcją `--good-names-rgxs="^[a-z]$"` albo `--good-names="x,y,z,"`. Tak samo nie przejmujcie się `too many arguments` albo `too many attributes`.
