# Applied AI: voorspellen met (bijna) elke dataset

Schoolproject voor Applied AI (NHL Stenden, 2026/2027). Opdrachtgever: X-Interactive (Ate) & Gerard van der Kolk.
Team: Gideon, Elmedin, Jesse, Lucas en Bram.

## Waar gaat het over?

Je gooit er een dataset in, het systeem zoekt zelf het patroon en voorspelt de uitkomst.
Het moet met alles werken: kassen, ziekteverzuim, machines, noem maar op. De opdrachtgever
vond **schaalbaar** en **modulair** het belangrijkst, dus:

- de core blijft altijd hetzelfde, een nieuwe use case is alleen een `manifest.json`
- het draait gewoon offline (één Docker-container, geen internet nodig)
- per use case een simpele baseline vs. LightGBM vs. een mix van de twee, de beste wint

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
- 515 tests, die draaien ook op GitHub Actions

## Datasets voor scenario-voorspelling

Voor de kwekers-use case (wat gebeurt er in week 44 als ik nu de kas van 20 naar 21 graden zet?) zijn er
testdatasets, zie [`docs/datasets.md`](docs/datasets.md):

- **gesimuleerde kwekerij** op echt KNMI-weer 2016-2025, met een *antwoordmodel*: voor elk scenario weten we de echte uitkomst, dus we kunnen testen of de voorspelling klopt
- **echte akkerbouwdata** (CBS + KNMI, 1999-2025), met een backtest: trainen tot 2020, testen op 2021-2025
- **echte kasdata** (Autonomous Greenhouse Challenge van WUR, 12 kassen, komkommer 2018 + tomaat 2019-20):
  stookenergie, lampstroom en **oogsttijd**. Twee eerlijke cijfers per model: voor een kas met eigen historie
  (stoken ±14% per week, twee keer zo goed als "zelfde als vorige week") en voor een kas die het model nooit zag
  (±32%, dus eerst een seizoen data verzamelen). Stroom ±2% per week, oogsttijd ±2,3 dagen
  (1 graad warmer = 2,9 dagen eerder). Zelf nalopen: `python scripts/check_greenhouse.py`
- **oogst maanden vooruit** (`tomato-season`): een gewasmodel dat is afgesteld en getest op de echte kassen
  (seizoenstotaal 3,7% ernaast op een kas die het nooit zag), gedraaid op 10 jaar echt KNMI-weer. Antwoord op
  "als ik nu mijn instellingen verander, wat oogst ik over 5 maanden?": 4% ernaast in de simulator. Nagespeeld op het echte seizoen 2019-20 van zes kassen: oogst over 3 maanden 8% ernaast, over 5 maanden 10%, stroom 5%, stoken 27% (`scripts/replay_real_season.py`).
  Eén scenario geeft ook het **nettoresultaat in euro's**: oogst min stook- en stroomkosten, met de eigen prijzen
  van de teler ("1 graad warmer: oogst +€0,79, stoken −€0,29, stroom −€0,25, netto +€0,26 per m²"). En het systeem zoekt zelf
  het **beste plan** (`/scenario/best`): alle instellingen doorrekenen en het plan met het hoogste nettoresultaat
  voorstellen. In de simulator levert het advies opvolgen +€6,09 per m² op (86% van wat achteraf maximaal kon)
- **orchidee in fases** (`orchid-batch`): een partij Phalaenopsis door opkweek, warme groei en koeling. Wanneer schuif je
  door en op welke temperatuur? Gebouwd op gepubliceerd teeltonderzoek (geen meetdata, dus een demonstratie). De
  planzoeker komt zelf uit op de gangbare praktijk: 8 weken koelen, op 20 °C (in de praktijk 19 °C)
- **eindtest op een ander jaar** (4e Greenhouse Challenge, 2024, nergens op afgesteld): de stook- en stroommodellen
  haalden het daar niet. Daarna gefixt: het stroommodel krijgt lampgebruik (uren × vermogen) en zit nu op 4% per seizoen;
  het stookmodel zegt bij een kas zonder historie eerlijk dat je er niet op moet varen, en buiten de trainingsdata
  komt er geen marge meer (zie [`docs/datasets.md`](docs/datasets.md))
- een kapotte versie van de kwekerijdata (vastgelopen sensor, pieken, verkeerde eenheden) voor later

```bash
pip install -e '.[datasets]' && python scripts/make_greenhouse_challenge.py   # echte kasdata (17 MB download)
python scripts/make_nursery.py --faults && python scripts/make_arable.py && python scripts/make_tomato_season.py && python scripts/make_orchid.py && predictor train
python scripts/check_scenarios.py   # klopt het model met het antwoordmodel?
```

In het manifest geef je aan aan welke knoppen (levers) de gebruiker mag draaien, binnen welk bereik,
en waar het altijd geldt welke kant op ("meer lamplicht = minder gas"). Daarna reken je scenario's door
(`POST /models/{id}/scenario`) of laat je één knop over het hele bereik lopen (`/scenario/sweep`). Je krijgt
het verschil met nu in eenheden, procenten en euro's, en of dat verschil groter is dan de foutmarge van het model:

> 1 degree warmer: gas use 20.6% lower (-5,210 m³, worth +€2,292). Based on only 7 comparable batches in the
> history; treat the size of the effect with some care.

Elk getal krijgt een marge waar de echte uitkomst in 80% van de gevallen binnen valt (gemeten op data die het
model niet zag, breder waar het model minder weet). Elk scenario zegt ook hoeveel vergelijkbare gevallen er in de historie zijn. Bij weinig vergelijkbare
gevallen is de voorspelling 3 tot 4 keer minder nauwkeurig, en dat zie je dan ook staan.

Zie [`docs/scenarios.md`](docs/scenarios.md).

## Meerdere datasets koppelen

Het weer en de eigenschappen van een kas hoeven niet meer in het CSV-bestand te zitten: in het manifest geef je onder
`joins` aan welke tabellen aan de basistabel hangen (op sleutel, op een tijdvenster, of allebei). Bijvoorbeeld "het gemiddelde
weer in de 10 weken na `start_date`":

```jsonc
"joins": [{ "name": "weather", "source": { "adapter": "csv-file", "path": "data/weather_rotterdam.csv" },
            "time": { "base": "start_date", "lookup": "date" },
            "window": { "start": { "offset": 0 }, "end": { "offset": 10, "unit": "weeks" } },
            "aggregate": { "outside_temp_avg_c": { "column": "temp_mean_c", "op": "mean" } } }]
```

- `predictor train` controleert de sleutels eerst (typfout, dubbele rijen, ontbrekende data) en `predictor check-joins` doet dat zonder te trainen
- de service vult de gekoppelde kolommen zelf in: stuur `start_date` en `compartment` mee in plaats van het weer; een waarde die je zelf meestuurt wint
- komt het venster buiten de data (volgend jaar), dan kan het manifest kiezen voor het gemiddelde van dezelfde periode in andere jaren, met een waarschuwing

Zie [`docs/technical.md`](docs/technical.md#joining-datasets).

De kwekerij is één use case, `nursery`, met vier uitkomsten (weken tot oogst, verkochte planten, gas, elektriciteit).
De oude ids (`nursery-gas`, `nursery-harvest-weeks`, `nursery-plants-sold`, `nursery-electricity`) werken nog als alias.

## Meerdere uitkomsten in één use case

Een use case kan meerdere dingen tegelijk voorspellen uit dezelfde invoer, bijvoorbeeld gas én verkochte planten:

- in het manifest staat `training.outputs` (naam → eenheid, grenzen, baseline, `algorithms`) in plaats van één `target`; `target` blijft werken voor één uitkomst
- invoer, joins, privacyregels en levers declareer je één keer; één `predictor train` geeft één modelversie met een apart model per uitkomst
- de uitkomsten zijn nooit elkaars features, dus er lekt niets van de ene uitkomst naar de andere
- per uitkomst kies je het model met `algorithms`: één naam forceert dat model, meerdere namen beperken de keuze, weglaten werkt als vanouds (baseline vs. LightGBM)
- `aliases` laten de oude use-case-ids doorwerken als losse use case met één uitkomst
- `/predict` geeft bij meerdere uitkomsten een `outputs`-blok per record; bij één uitkomst is het antwoord ongewijzigd
- het dashboard toont een chip per uitkomst en een lijst met alle uitkomsten

```jsonc
{
  "use_case_id": "nursery",
  "training": {
    "time_column": "start_date",
    "drop_columns": ["batch_id", "compartment"],
    "outputs": {
      "weeks_to_harvest": { "unit": "weeks" },
      "gas_m3": { "unit": "m³", "target_min": 0, "algorithms": ["lightgbm"] },       // geforceerd
      "electricity_kwh": { "unit": "kWh", "target_min": 0, "algorithms": ["blend", "lightgbm"] }   // keuze uit twee
    }
  },
  "model": { "runner": "onnx", "version": "latest" },
  "aliases": { "nursery-gas": "gas_m3", "nursery-electricity": "electricity_kwh" }
}
```

Zie [`docs/technical.md`](docs/technical.md#several-outputs).

## Nog te doen

- tijdreeksen met historie (lags/rolling features)
- login/auth voordat het echt online gaat
