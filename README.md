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
| [`prototype/`](prototype/) | De werkende backend (Python, FastAPI, ONNX). Daar staat ook de uitgebreide README. |

Later komt er waarschijnlijk meer bij (bijv. een dashboard), dat krijgt dan een eigen map.

## Snel proberen

```bash
cd prototype
docker build -t applied-ai-predictor .
docker run --rm -p 8000:8000 applied-ai-predictor
```

Open daarna http://localhost:8000/docs, of draai `./scripts/demo.sh` voor de demo.

Zonder Docker kan ook, zie [`prototype/README.md`](prototype/README.md).

## Wat kan het nu

- 4 use cases: kas-temperatuur over een uur, ziekteverzuim, machine-uitval (echte AI4I-data)
  en afwijkingen spotten zonder labels
- nieuwe dataset toevoegen: `predictor init` → `predictor train` → klaar, zonder code aan te passen
- privacyregels per use case (bijv. geen `diagnosis`-veld, geen groepjes kleiner dan 5)
- 52 tests, die draaien ook op GitHub Actions

## Nog te doen

- tijdreeksen met historie (lags/rolling features)
- login/auth voordat het echt online gaat
- dashboard (nice to have)
