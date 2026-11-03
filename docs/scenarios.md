# Scenario prediction

The goal (meeting 29-10): a grower changes an input, for example the greenhouse temperature from
20 to 21 °C, and sees what that does to the outcome, for example the number of plants and the gas
used. This document grows with the scenario tickets.

| Ticket | Status |
|---|---|
| 1. Datasets for testing | done, see [datasets.md](datasets.md) |
| 2. Scenario variables in manifest.json | **done (this page)** |
| 3. Scenario input | next |
| 4. Calculate a scenario | |
| 5. Difference base vs. scenario | |

## Scenario variables in `manifest.json` (ticket 2)

A use case declares which inputs a user may change (**levers**), within which range, and, only
where it always holds, in which direction a lever moves an output:

```jsonc
"scenario": {
  "levers": {
    "temp_setpoint_c": { "label": "Greenhouse temperature", "unit": "°C", "min": 17, "max": 23, "step": 0.5 },
    "light_hours": {
      "label": "Lamp hours", "unit": "h/day", "min": 0, "max": 18, "step": 0.5,
      "effects": { "gas_m3": "decreasing" }     // more lamp light never costs more gas (lamp heat)
    }
  },
  "outputs": {
    "gas_m3": { "label": "Gas use", "value_eur": -0.45 }   // positive = you sell it, negative = you pay for it
  }
}
```

Everything that is not a lever is **context** (weather, season, compartment): a scenario never changes it.

### What is checked, and when

| When | Check |
|---|---|
| Loading the manifest | `min` < `max`, `step` > 0, effects are `increasing`/`decreasing`, effects and outputs name a real output, a lever is not the target, time column, a dropped column or a forbidden (privacy) field, scenario only for use cases with a target, typos rejected |
| Training | every lever is a numeric or yes/no model input (not missing, constant or categorical) |
| Starting the API | every lever is an input of the *trained* model; a yes/no lever has range 0-1 step 1 |

### Declared effects are hard rules for the model

`effects` become **monotone constraints** in LightGBM: whatever the situation, turning that lever
up can only move the output the declared way. A linear model cannot be constrained, so it is
checked instead: if it contradicts a declared effect it is not eligible to win. The training report
lists the levers, the constraints, and any model that was excluded.

**Only declare what always holds.** In the simulated nursery, "more heating = more gas" is *not*
true per batch: a warmer batch is ready sooner, and 14% of batches use less gas in total. So
temperature is a lever without a declared effect, while lamp light, CO₂ and plant density do have
one (`tests/test_scenario_variables.py` checks every declared effect against the simulator's answer key).

This already matters in the nursery gas model: the linear baseline learned "more CO₂ = more gas",
because growers dose CO₂ in winter, exactly when they heat. That is a correlation, not an effect, so
the linear model was excluded and the constrained LightGBM won.

The result on test batches the models never saw:

![Before and after declaring effects](img/scenario_constraints.png)

- plant density → gas had the **wrong sign** (+1.3% instead of −3.0%); now −3.3%
- CO₂ → weeks to harvest: right direction for 96% of batches (was 78%), the rest "no change", never the opposite
- accuracy did not suffer: harvest-weeks R² 0.749 → 0.761, gas MAE 3,230 → 3,174 m³

Indirect effects (lamp heat saving gas) are still underestimated; better features for that are
input for ticket 4.

### What clients get: `GET /models/{id}` → `scenario`

```json
{
  "levers": [
    { "name": "light_hours", "label": "Lamp hours", "unit": "h/day", "kind": "number",
      "min": 0, "max": 18, "step": 0.5, "effects": { "gas_m3": "decreasing" },
      "training_range": [0, 18], "beyond_training_data": false }
  ],
  "outputs": [ { "name": "gas_m3", "label": "Gas use", "unit": "m³", "value_eur": -0.45 } ],
  "constraints": { "effects": { "light_hours": "decreasing", "co2_ppm": "decreasing", "plant_density_m2": "decreasing" },
                   "excluded_candidates": { "linear_regression": ["co2_ppm should be decreasing but the model says increasing"] } }
}
```

`beyond_training_data` warns when a lever's range goes past what the model has seen: tree models
predict a flat line there, so a scenario at the edge looks like "no effect".

### Not in this ticket

- Several outputs in one use case (the multi-output tickets): `outputs` and `effects` are already
  keyed by output name, so they work unchanged once a use case has more than one target.
- Sending a scenario and calculating it (tickets 3 and 4), money and the difference view (ticket 5).
