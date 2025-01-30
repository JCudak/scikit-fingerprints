# Testowanie wpływu filtrów molekularnych na wyniki klasyfikacji cząstek

### Autorzy: Jakub Cudak, Karol Augustyniak



Wykorzystano bibliotekę scikit-fingerprints

# Struktura projektu 

## ```config.py```

Zawiera wszystkie zbadane filtry molekularne oraz fingerprinty.

## ```dataset_processor.py```

Zawiera klasę `DatasetProcessor`, która dzieli dane na zbiory treningowy, walidacyjny i testowy oraz umożliwia zastosowanie filtrów przetwarzania.

Konstruktor ```DatasetProcessor(dataset_name, data, labels)```

- dataset_name (str): Nazwa zestawu danych.
- data (iterable): Dane wejściowe.
- labels (iterable): Etykiety danych.

Metoda ```get_filtered_data(filter_fn=None)```:

Zwraca zbiory danych (X_train, y_train, X_valid, y_valid, X_test, y_test) przepuszcone przed podany filtr molekularny.


## ```evaluation.py```

Funkcja ```check_filters```

Testuje pojedyncze filtry pod kątem ich wpływu na dokładność modelu dla różnych fingerprintów.

Argumenty:

- processor: Obiekt przetwarzający dane.
- model_pipeline: Pipeline modelu klasyfikacyjnego.
- dataset_name: Nazwa zbioru danych.
- results: Lista wyników eksperymentów.

Funkcja ```check_combinations```

Testuje dwuelementowe kombinacje filtrów w celu oceny ich wpływu na dokładność modelu dla różnych fingerprintów molekularnych.

Argumenty:

- processor: Obiekt przetwarzający dane.
- model_pipeline: Pipeline modelu klasyfikacyjnego.
- dataset_name: Nazwa zbioru danych.
- results: Lista wyników eksperymentów.

Działanie:

Oblicza dokładność bazową modelu na danych bez filtrów.
Iteruje przez kombinacje filtrów, przetwarza dane i oblicza dokładność oraz różnicę względem bazy.
Wyniki zapisuje w results i wyświetla.


## ```model_pipeline.py```

Klasa ```ModelPipeline``` odpowiada za przetwarzanie danych wejściowych: konwersję ich na fingerprinty molekularne oraz trenowanie i ocenę modelu klasyfikacyjnego.

Metoda ```process(self, train_X, train_y, test_X, test_y, fingerprint_class)```

Przetwarza dane wejściowe i trenuje model:

- Konwertuje dane SMILES na fingerprinty za pomocą smiles_to_fingerprint.
- Tworzy model klasyfikacyjny przy użyciu fabryki modeli.
- Trenuje model na danych treningowych.
- Przewiduje etykiety dla danych testowych i zwraca dokładność (średnia zgodność predykcji z rzeczywistymi etykietami).

Metoda ```create_model```:

Tworzy instancję klasyfikatora Random Forest z ustalonymi hiperparametrami.

Zwraca: Obiekt RandomForestClassifier.

## ```results.py```

Zawiera funkcje pomocnicze do zapisu wyników.

Funkcja ```save_results_to_json```:

Zapisuje wynik do pliku JSON o podanej nazwie

Funkcja ```calculate_accuracy_difference```:

zwraca różnicę dokładności modelu z zastosowanymi filtrami a dokładnością bazową

Funkcja ```print_results```:

Wypisuje sformatowany tekst przedstawiający wyniki działania modelu

## ```utils.py```

Funkcja ```get_data_and_labels_at```:

Zwraca wybrane dane oraz ich etykiety

Funkcja ```smiles_to_fingerprint```:

Konwertuje dane w formacie SMILES na wybrany fingerprint molekularny

Funkcja ```activate_filter```:

Zwraca przefiltrowane podzbiory danych i etykiet

Funkcja ```filter_x_and_y```:

Filtruje dane i odpowiadające im etykiety za pomocą przesłanego filtra


## ```main.py```

Skrypt wykonuje ocenę modeli klasyfikacyjnych na różnych zbiorach danych z zastosowaniem różnych filtrów i ich kombinacji. Wyniki są zapisywane w plikach JSON.