# Quantum Scheduling

Projekt demonstruje rozwiązywanie problemu **job-shop scheduling**: każde zadanie składa się z uporządkowanych operacji, a operacja wymaga wskazanej maszyny przez określony czas. Implementacja udostępnia dwa podejścia:

- klasyczny solver CP-SAT z Google OR-Tools, służący jako punkt odniesienia;
- model QUBO rozwiązywany przez symulację QAOA w Qiskit.

Wynikiem jest harmonogram operacji oraz jego `makespan` — czas zakończenia ostatniej operacji.

> Stan repozytorium: aktywna implementacja znajduje się w pakiecie [`jobshop/`](jobshop). Katalog [`controlled code/`](controlled%20code) zawiera modele I/O, przykładowe dane, wyniki i narzędzie do wykresów, ale jego testy odwołują się do nieobecnego modułu `algorithms`. Szczegóły w sekcji [Znane niespójności](#znane-niespójności).

## Model problemu

Dla operacji `k` zadania `j` o czasie trwania `d[j, k]` model QUBO używa zmiennej binarnej:

```text
x[j, k, t] = 1  ⇔  operacja k zadania j rozpoczyna się w chwili t
```

Przed utworzeniem zmiennych obliczane są najwcześniejsze i najpóźniejsze dopuszczalne starty (`ES`, `LS`), więc nie powstają zmienne poza oknem `[ES, LS]`. Macierz QUBO zawiera:

1. karę one-hot, wymuszającą dokładnie jeden start każdej operacji;
2. karę za naruszenie kolejności operacji w tym samym zadaniu;
3. karę za nakładające się operacje na tej samej maszynie;
4. koszt równy sumie czasów zakończenia ostatnich operacji zadań.

Całkowita energia ma postać `H(x) = H_cost(x) + λ H_penalty(x)`. Gdy `penalty_lambda` nie zostanie podane, `jobshop.qubo_builder.calculate_sufficient_lambda()` dobiera zachowawczą wagę kary na podstawie współczynników funkcji kosztu.

Schemat przepływu:

```text
instancja / InstanceData
          │
          ├── CP-SAT ────────────────► harmonogram referencyjny
          │
          └── QUBO builder ─► QAOA ─► dekoder ─► CalculationState
```

## Struktura projektu

| Ścieżka | Rola |
| --- | --- |
| [`jobshop/models.py`](jobshop/models.py) | Encje domenowe, gotowe instancje 2×2 i 3×3 oraz loader JSON. |
| [`jobshop/qubo_builder.py`](jobshop/qubo_builder.py) | Okna startów, budowa macierzy QUBO i dobór `λ`. |
| [`jobshop/qubo_decoder.py`](jobshop/qubo_decoder.py) | Zamiana bitów QUBO na `CalculationState`. |
| [`jobshop/cp_sat_solver.py`](jobshop/cp_sat_solver.py) | Dokładny solver klasyczny OR-Tools CP-SAT. |
| [`jobshop/qaoa_solver.py`](jobshop/qaoa_solver.py) | Konwersja QUBO→Ising i QAOA na symulatorze statevector. |
| [`jobshop/compare_solvers.py`](jobshop/compare_solvers.py) | Uruchomienie i zestawienie CP-SAT z QAOA. |
| [`controlled code/load.py`](controlled%20code/load.py), [`save.py`](controlled%20code/save.py) | Wczytywanie instancji i zapis harmonogramu JSON. |
| [`controlled code/plot_results.py`](controlled%20code/plot_results.py) | Generowanie wykresów Gantta z plików wynikowych. |
| [`controlled code/resources/`](controlled%20code/resources) | Instancje 2×2, 3×3, format wyniku, przykładowe harmonogramy i wykresy. |

## Wymagania i instalacja

Wymagany jest Python 3.11 lub nowszy. Zależności są zdefiniowane w [`pyproject.toml`](pyproject.toml), w tym NumPy, SciPy, OR-Tools, Qiskit, Matplotlib i opcjonalny SDK VeloxQ.

Najprościej użyć `uv`:

```bash
uv sync
```

Alternatywnie należy zainstalować zależności wymienione w `pyproject.toml` w wybranym środowisku Pythona. Projekt nie definiuje obecnie skryptów CLI ani celu `make`.

## Uruchomienie porównania solverów

Po instalacji zależności uruchom porównanie dla domyślnej instancji 3×3:

```bash
uv run python -m jobshop.compare_solvers
```

Wariant 2×2:

```bash
uv run python -m jobshop.compare_solvers 2x2
```

Skrypt kolejno:

1. wyznacza rozwiązanie CP-SAT;
2. buduje QUBO i podaje liczbę zmiennych binarnych;
3. uruchamia jednowarstwowe QAOA (`reps=1`, maksymalnie 100 iteracji COBYLA);
4. dekoduje bity do harmonogramu i wypisuje porównanie `makespan` oraz czasu działania.

QAOA używa pełnego wektora stanu i enumeruje `2^N` stanów przy obliczaniu energii. Jest to demonstracja przeznaczona dla małych QUBO, nie metoda dla dużych instancji produkcyjnych. Do klasycznego, dokładnego rozwiązania użyj `solve_cp_sat`.

Przykład użycia API:

```python
from jobshop import (
    build_job_shop_qubo,
    decode_qubo_solution,
    make_test_instance,
    solve_cp_sat,
    solve_with_qaoa,
)

instance = make_test_instance("2x2")

# Klasyczne rozwiązanie referencyjne
cp_result = solve_cp_sat(instance)
print(cp_result["makespan"])

# Ścieżka QUBO + QAOA
Q, metadata = build_job_shop_qubo(instance)
result = solve_with_qaoa(Q, offset=metadata["offset"], reps=1, maxiter=100)
schedule = decode_qubo_solution(result["bits"], metadata)
print(schedule.makespan, schedule.metadata["status"])
```

## Format danych

Instancja JSON zawiera problem oraz listę zadań z uporządkowanymi operacjami:

```json
{
  "problem": "erp_scheduling",
  "jobs": [
    {
      "id": "Z1",
      "operations": [
        { "machine": "T", "duration": 2 },
        { "machine": "F", "duration": 1 }
      ]
    }
  ]
}
```

Pliki przykładowe to [`instance_2x2.json`](controlled%20code/resources/instance_2x2.json) i [`instance_3x3.json`](controlled%20code/resources/instance_3x3.json). Wynik jest zapisywany jako:

```json
{
  "instance": "instance_2x2",
  "method": "Nazwa metody",
  "makespan": 4,
  "schedule": [
    { "job": "Z2", "machine": "T", "start": 0, "end": 1 }
  ]
}
```

Pełny wzorzec znajduje się w [`output_format.json`](controlled%20code/resources/output_format.json). `CalculationState.validate()` sprawdza brak nakładania się operacji na maszynach i liczbę operacji względem instancji.

## Wykres Gantta

Z istniejących wyników można wygenerować wykresy PNG:

```bash
uv run python "controlled code/plot_results.py"
```

Bez argumentów skrypt przetwarza wszystkie `controlled code/resources/result_*.json` i zapisuje obrazy w `controlled code/resources/plots/`. Można przekazać własny plik oraz katalog wyjściowy:

```bash
uv run python "controlled code/plot_results.py" \
  "controlled code/resources/result_instance_2x2.json" \
  --output-dir /tmp/jobshop-plots
```

## Znane niespójności

- `pyproject.toml` wskazuje `testpaths = ["tests"]`, lecz taki katalog nie występuje w repozytorium.
- Istniejące testy w `controlled code/Tests/` importują `algorithms.qubo`, `algorithms.sufficient_lambda` i `algorithms.velox_solver`. Katalog `controlled code/algorithms/` nie jest obecny, więc te testy oraz `controlled code/test_velox.py` nie uruchomią się w aktualnym drzewie plików.
- Integracja VeloxQ widoczna jedynie w tym osieroconym kodzie nie jest obecnie połączona z pakietem `jobshop/`. Zmienna `VELOX_TOKEN` byłaby potrzebna dopiero po przywróceniu tej integracji; nie zapisuj tokenu w repozytorium.
- [`LOG.md`](LOG.md) opisuje wcześniejszą architekturę (`anneal.py`, `bruteforce.py`, `scheduler.py`), której nie ma w obecnym kodzie, i nie jest źródłem dokumentacji bieżącej implementacji.

## Licencja

MIT — zobacz [`LICENSE`](LICENSE).
