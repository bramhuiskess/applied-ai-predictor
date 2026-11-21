"""A pot-orchid (Phalaenopsis) batch through its phases, built on published growing research.

There is no open orchid dataset, so unlike the tomato crop model this one is **not fitted on measurements**.
Every relation below comes from published research or common Dutch practice, and is marked as such; where a
number is our own assumption it says so. It shows how the product handles a crop with phases. A grower's own
batch records would replace these numbers.

The batch, as the grower described it:

1. **Phase 1, raising**: young plants, close together, warm.
2. **Phase 2, warm growth**: wider apart, warm, so the plant makes leaves and stays vegetative.
3. **Phase 3, cooling**: cool, so the plant makes flower spikes. Then it is finished at 21 C until sale.

What is known, and where from:

* Growers keep the warm phases at a constant 28 C to keep plants vegetative; 31 C gives more leaves
  (Wageningen UR reports; Lee et al. 2021).
* A day temperature of 29 C or more prevents flowering, 26 C or less induces it (Blanchard & Runkle 2006;
  Newton & Runkle 2009). So a warm phase below about 27 C gives early spikes that have to be removed.
* Cooling is about 8 weeks at 19 C, then about 10 weeks at 21 C to sale; 8 weeks of cooling always gave more
  plants with several spikes than shorter cooling; 22 C can replace 19 C in part (Wageningen UR / Kas als
  Energiebron reports).
* The number of spikes follows the leaves made in the warm phases. With 3, 4 and 5 mol of light per day in the
  warm phases a trial counted 1.88, 1.92 and 2.03 spikes per plant (Wageningen UR, Teeltversnelling Phalaenopsis).

Our own assumptions (no source): the leaf rate of 0.2 per week at 28 C, the shape of the curves between the
published points, the plant densities per phase, and the base temperature of spike development.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import greenhouse_heat as heating

STANDARD = {"weeks_phase_1": 16, "weeks_phase_2": 12, "temp_warm_c": 28.0, "weeks_cooling": 8, "temp_cooling_c": 19.0}
FINISH_TEMP_C = 21.0
START_LEAVES = 2.0                 # a young plant from the laboratory
LEAVES_PER_WEEK = 0.20             # at 28 C and 4 mol light (assumption)
SPIKE_BASE_TEMP_C = 10.0           # spike development stops below this (assumption)
SPIKE_DEGREE_DAYS = 8 * 7 * (19 - 10) + 10 * 7 * (21 - 10)   # 8 weeks at 19 C + 10 weeks at 21 C (practice)
PLANTS_PER_M2 = {1: 90.0, 2: 50.0, 3: 45.0}                  # per phase (assumption, 12 cm pots)
COOLING_COP = 3.0                  # heat removed per unit of electricity (assumption)
COOLING_SUN_SHARE = 0.25           # share of the outside radiation that has to be cooled away in a shaded orchid house
                                   # (assumption; the tomato heat balance only knows how much sun saves on heating)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Batch:
    spikes_per_plant: float        # spikes per plant put in, the average of the batch
    sellable_share: float          # share of the plants that flower and can be sold (0-1)
    two_spike_share: float         # share of the plants with two or more spikes: the better-paid class (0-1)
    weeks_to_sale: float
    weeks_finishing: float
    leaves_at_cooling: float
    early_spike_share_pct: float   # plants that flowered too early in the warm phases


def grow(weeks_phase_1: float, weeks_phase_2: float, temp_warm_c: float, weeks_cooling: float, temp_cooling_c: float,
         light_mol: float = 4.0) -> Batch:
    """The plant side of a batch: leaves in the warm phases, spikes in the cooling, time to sale."""
    warm_weeks = weeks_phase_1 + weeks_phase_2
    # Leaves: faster when warmer (more leaves at 31 than at 28 C) and with more light.
    temp_factor = float(np.clip(1.0 + 0.06 * (temp_warm_c - 28.0), 0.4, 1.25))
    light_factor = (light_mol / 4.0) ** 0.25
    leaves = START_LEAVES + LEAVES_PER_WEEK * temp_factor * light_factor * warm_weeks

    # Below about 27 C the plant already makes spikes in the warm phase; they are removed and cost later spikes.
    early = float(np.clip((27.0 - temp_warm_c) * 0.25, 0.0, 0.6))
    heat_stress = float(np.clip(1.0 - 0.05 * (temp_warm_c - 30.0), 0.7, 1.0))

    # Induction: at 26 C or less every plant makes a spike, at 29 C none.
    induced = float(np.clip((28.5 - temp_cooling_c) / 3.0, 0.0, 1.0))
    # Enough weeks of cooling: 8 weeks is always best, short cooling gives few second spikes.
    duration = float(np.clip((weeks_cooling - 2.0) / 6.0, 0.0, 1.0)) ** 0.7
    induced *= float(np.clip(weeks_cooling / 4.0, 0.0, 1.0))
    # A second spike asks for a big enough plant and real cooling (19-20 C best, 22 C less).
    cool = float(np.clip(1.0 - 0.125 * (temp_cooling_c - 20.0), 0.0, 1.0))
    second = _sigmoid((leaves - 6.3) / 0.7) * cool * duration * heat_stress
    third = 0.25 * _sigmoid((leaves - 8.2) / 0.5) * cool * duration * heat_stress
    spikes = induced * (1.0 + second + third) - 0.3 * early * induced
    two_spike_share = induced * second * (1.0 - 0.3 * early)

    # Spike development is a temperature sum: cooler or shorter cooling means longer finishing at 21 C.
    done = weeks_cooling * 7 * max(temp_cooling_c - SPIKE_BASE_TEMP_C, 0.0)
    finishing = max(4.0, (SPIKE_DEGREE_DAYS - done) / (7 * (FINISH_TEMP_C - SPIKE_BASE_TEMP_C)))
    return Batch(round(float(max(spikes, 0.0)), 3), round(float(induced), 3), round(float(two_spike_share), 3),
                 round(float(warm_weeks + weeks_cooling + finishing), 1), round(float(finishing), 1), round(float(leaves), 2),
                 round(float(100 * early), 1))


def energy(outside_c: np.ndarray, radiation_j_cm2: np.ndarray, weeks_phase_1: float, weeks_phase_2: float, temp_warm_c: float,
           weeks_cooling: float, temp_cooling_c: float, weeks_finishing: float) -> dict[str, float]:
    """Heating (MJ), cooling electricity (kWh) and space (m2-weeks) per plant, on the batch's own weather.

    Uses the heat balance fitted on the tomato greenhouses (`greenhouse_heat.py`) for heating; when
    the sun brings in more than leaks out, the greenhouse has to be cooled to hold the temperature, which costs electricity. The weather arrays start
    on the first day of the batch.
    """
    p = heating.PARAMETERS
    result = {"heat_mj_per_plant": 0.0, "cooling_kwh_per_plant": 0.0, "space_m2_weeks_per_plant": 0.0}
    day = 0
    for phase, weeks, temp in [(1, weeks_phase_1, temp_warm_c), (2, weeks_phase_2, temp_warm_c),
                               (3, weeks_cooling, temp_cooling_c), (3, weeks_finishing, FINISH_TEMP_C)]:
        days = int(round(weeks * 7))
        outside, radiation = outside_c[day: day + days], radiation_j_cm2[day: day + days]
        balance = p.loss_mj_per_degree * (temp - outside) - p.sun_share * radiation / 100.0 + p.base_mj
        # Cooling: the sun that comes in, minus what leaks out to a colder outside.
        load = COOLING_SUN_SHARE * radiation / 100.0 - p.loss_mj_per_degree * (temp - outside)
        area = 1.0 / PLANTS_PER_M2[phase]
        result["heat_mj_per_plant"] += float(np.clip(balance, 0, None).sum()) * area
        result["cooling_kwh_per_plant"] += float(np.clip(load, 0, None).sum()) / 3.6 / COOLING_COP * area
        result["space_m2_weeks_per_plant"] += weeks * area
        day += days
    return result
