# Applied AI: voorspellen met (bijna) elke dataset

Schoolproject voor Applied AI (NHL Stenden, 2026/2027). Opdrachtgever: X-Interactive (Ate) & Gerard van der Kolk.
Team: Gideon, Elmedin, Jesse, Lucas en Bram.

## Waar gaat het over?

Je gooit er een dataset in, het systeem zoekt zelf het patroon en voorspelt de uitkomst.
Het moet met alles werken: kassen, ziekteverzuim, machines, noem maar op. De opdrachtgever
vond **schaalbaar** en **modulair** het belangrijkst, dus:

- de core blijft altijd hetzelfde, een nieuwe use case is alleen een `manifest.json`
- het draait gewoon offline (één Docker-container, geen internet nodig)
- per use case een simpele baseline vs. LightGBM, de beste wint

## Wat zit waar

| Map | Wat |
|---|---|
| [`predictor/`](predictor/) | De backend zelf: core (API, validatie, privacy), adapters (ONNX, opslag) en de trainer |
| [`usecases/`](usecases/) | Eén `manifest.json` per use case, verder geen code |
| [`scripts/`](scripts/) | Datasets maken/downloaden, demo en benchmark |
| [`tests/`](tests/) | Alle tests (draaien ook op GitHub Actions) |
| [`notebooks/`](notebooks/) | Technische uitleg van de modellen |
| [`docs/`](docs/) | Uitgebreide technische docs ([`technical.md`](docs/technical.md)) |

## Snel proberen

```bash
docker build -t applied-ai-predictor .
docker run --rm -p 8000:8000 applied-ai-predictor
```

Open daarna http://localhost:8000 voor het dashboard (grafieken + live voorspellen),
http://localhost:8000/docs voor de API, of draai `./scripts/demo.sh` voor de demo in de terminal.

Zonder Docker kan ook (`./scripts/setup.sh`, vereist Python 3.11+), zie [`docs/technical.md`](docs/technical.md).

## Wat kan het nu

- 4 use cases: kas-temperatuur over een uur, ziekteverzuim, machine-uitval (echte AI4I-data)
  en afwijkingen spotten zonder labels
- nieuwe dataset toevoegen: `predictor init` → `predictor train` → klaar, zonder code aan te passen
- privacyregels per use case (bijv. geen `diagnosis`-veld, geen groepjes kleiner dan 5)
- dashboard met grafieken per use case en een "probeer het zelf"-paneel met de JSON erbij
- 191 tests, die draaien ook op GitHub Actions

## Datasets voor scenario-voorspelling

Voor de kwekers-use case (wat gebeurt er in week 44 als ik nu de kas van 20 naar 21 graden zet?) zijn er
testdatasets, zie [`docs/datasets.md`](docs/datasets.md):

- **gesimuleerde kwekerij** op echt KNMI-weer 2016-2025, met een *antwoordmodel*: voor elk scenario weten we de echte uitkomst, dus we kunnen testen of de voorspelling klopt
- **echte akkerbouwdata** (CBS + KNMI, 1999-2025), met een backtest: trainen tot 2020, testen op 2021-2025
- **echte kasdata** (Autonomous Greenhouse Challenge van WUR, 6 kassen, dec 2019-mei 2020): stookenergie en
  lampstroom per dag, getest op een kas die het model nooit gezien heeft (stroom R² 0.98, stoken R² 0.57)
- een kapotte versie van de kwekerijdata (vastgelopen sensor, pieken, verkeerde eenheden) voor later

```bash
pip install -e '.[datasets]' && python scripts/make_greenhouse_challenge.py   # echte kasdata (8 MB download)
python scripts/make_nursery.py --faults && python scripts/make_arable.py && predictor train
python scripts/check_scenarios.py   # klopt het model met het antwoordmodel?
```

In het manifest geef je aan aan welke knoppen (levers) de gebruiker mag draaien, binnen welk bereik,
en waar het altijd geldt welke kant op ("meer lamplicht = minder gas"). Daarna reken je scenario's door
(`POST /models/{id}/scenario`) of laat je één knop over het hele bereik lopen (`/scenario/sweep`). Je krijgt
het verschil met nu in eenheden, procenten en euro's, en of dat verschil groter is dan de foutmarge van het model:

> 1 degree warmer: gas use 20.6% lower (-5,210 m³, worth +€2,292). Based on only 7 comparable batches in the
> history; treat the size of the effect with some care.

Elk scenario zegt ook hoeveel vergelijkbare gevallen er in de historie zijn. Bij weinig vergelijkbare
gevallen is de voorspelling 3 tot 4 keer minder nauwkeurig, en dat zie je dan ook staan.

Zie [`docs/scenarios.md`](docs/scenarios.md).

## Nog te doen

- tijdreeksen met historie (lags/rolling features)
- login/auth voordat het echt online gaat
