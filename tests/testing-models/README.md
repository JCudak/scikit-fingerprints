# Testowanie wpływu filtrów molekularnych na wyniki klasyfikacji cząstek

### Autorzy: Jakub Cudak, Karol Augustyniak



Wykorzystano bibliotekę scikit-fingerprints

# Wyniki 


## ```filter_results.json```

Zawiera wyniki modelu dla wszystkich filtrów na zbiorach danych.
Wykorzystuje 'AtomPairFingerprint'

#### Dataset 'BACE'
| dataset_name | filter_name      | filter_accuracy | filter_difference |
|--------------|------------------|-----------------|-------------------|
| BACE         | rule_of_3        | 0.9000          | 0.2618            |
| BACE         | faf4_druglike    | 0.8378          | 0.1997            |
| BACE         | faf4_leadlike    | 0.8261          | 0.1879            |

---

#### Dataset 'BBBP'
| dataset_name | filter_name           | filter_accuracy | filter_difference |
|--------------|-----------------------|-----------------|-------------------|
| BBBP         | rule_of_3             | 0.7500          | 0.1863            |
| BBBP         | tice_insecticides     | 0.6549          | 0.0911            |
| BBBP         | hao                   | 0.6525          | 0.0888            |

---

#### Dataset 'ClinTox'
| dataset_name | filter_name         | filter_accuracy | filter_difference |
|--------------|---------------------|-----------------|-------------------|
| ClinTox      | rule_of_3           | 1.0000          | 0.0642            |
| ClinTox      | tice_herebicides    | 0.9593          | 0.0235            |
| ClinTox      | lint                | 0.9588          | 0.0230            |

---

#### Dataset 'HIV'
| dataset_name | filter_name         | filter_accuracy | filter_difference |
|--------------|---------------------|-----------------|-------------------|
| HIV          | rule_of_3           | 0.9859          | 0.0160            |
| HIV          | tice_herebicides    | 0.9852          | 0.0154            |
| HIV          | faf4_druglike       | 0.9848          | 0.0150            |

## ```filter_results2.json```

Zawiera wyniki modelu dla kombinacji par najlepszych filtrów na zbiorach danych
Wykorzystuje 'AtomPairFingerprint'

#### Dataset 'BACE'
| dataset_name | filter_name                          | filter_accuracy | filter_difference |
|--------------|--------------------------------------|-----------------|-------------------|
| BACE         | [rule_of_3, faf4_druglike]          | 1.0000          | 0.3618            |
| BACE         | [rule_of_3, faf4_leadlike]          | 1.0000          | 0.3618            |
| BACE         | [rule_of_3, tice_herebicides]       | 1.0000          | 0.3618            |

---

#### Dataset 'BBBP'
| dataset_name | filter_name                          | filter_accuracy | filter_difference |
|--------------|--------------------------------------|-----------------|-------------------|
| BBBP         | [rule_of_3, faf4_leadlike]          | 0.7500          | 0.1863            |
| BBBP         | [rule_of_3, tice_insecticides]      | 0.7500          | 0.1863            |
| BBBP         | [rule_of_3, hao]                    | 0.7500          | 0.1863            |

---

#### Dataset 'ClinTox'
| dataset_name | filter_name                          | filter_accuracy | filter_difference |
|--------------|--------------------------------------|-----------------|-------------------|
| ClinTox      | [rule_of_3, faf4_druglike]          | 1.0000          | 0.0642            |
| ClinTox      | [rule_of_3, faf4_leadlike]          | 1.0000          | 0.0642            |
| ClinTox      | [rule_of_3, tice_insecticides]      | 1.0000          | 0.0642            |

---

#### Dataset 'HIV'
| dataset_name | filter_name                          | filter_accuracy | filter_difference |
|--------------|--------------------------------------|-----------------|-------------------|
| HIV          | [rule_of_3, faf4_druglike]          | 0.9915          | 0.0214            |
| HIV          | [rule_of_3, faf4_leadlike]          | 0.9891          | 0.0190            |
| HIV          | [faf4_druglike, tice_herebicides]   | 0.9884          | 0.0183            |

---

#### Dataset 'SIDER'
| dataset_name | filter_name                          | filter_accuracy | filter_difference |
|--------------|--------------------------------------|-----------------|-------------------|
| SIDER        | [rule_of_3, faf4_druglike]          | 0.8148          | 0.0311            |
| SIDER        | [faf4_leadlike, lint]               | 0.8122          | 0.0284            |
| SIDER        | [rule_of_3, lint]                   | 0.8056          | 0.0218 


## ```combination_results.json``` 
- Zawiera wyniki klasyfikacji modelu na przetworzonych zbiorach danych.
- Preprocessing danych wykonany jest za pomocą kombinacji dwóch filtrów molekularnych oraz zbioru różnych fingerprintów

#### Dataset 'BACE'
| dataset_name | fingerprint_name | filter_name                          | filter_accuracy | filter_difference |
|--------------|------------------|--------------------------------------|-----------------|-------------------|
| BACE         | MAPFingerprint   | [rule_of_3, faf4_druglike]          | 1.0000          | 0.4872            |
| BACE         | MAPFingerprint   | [rule_of_3, faf4_leadlike]          | 1.0000          | 0.4872            |
| BACE         | MAPFingerprint   | [rule_of_3, tice_herebicides]       | 1.0000          | 0.4872            |

---

#### Dataset 'BBBP'
| dataset_name | fingerprint_name | filter_name                          | filter_accuracy | filter_difference |
|--------------|------------------|--------------------------------------|-----------------|-------------------|
| BBBP         | MAPFingerprint   | [rule_of_3, faf4_leadlike]          | 0.7500          | 0.2412            |
| BBBP         | MAPFingerprint   | [rule_of_3, tice_insecticides]      | 0.7500          | 0.2412            |
| BBBP         | MAPFingerprint   | [rule_of_3, hao]                    | 0.7500          | 0.2412            |

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