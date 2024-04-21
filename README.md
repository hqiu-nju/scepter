# SCEPTer
**S**imulating **C**onstellation **E**mission **P**atterns for **Te**lescopes (**r**adio)

This is a repository to simulate the satellite constellation emissions and measure the EPFD of the observed sky area.

The simulation can be performed using a simulated constellation or from real satellite constellation two line elements (TLEs).
Satellite TLEs can be found on https://celestrak.org/

We use the PyCRAF and cysgp4 packages for many of the base calculations, see requirements.txt for list of dependancies, code was written and tested in Python 3.10

## Running the observation simulator

In the pfd module, we have the tools for creating an observation simulation that will provide an RFI sky model from satellite emissions.

The simulation is organised through multiple dimensions to give consideration of the telescope pointings, satellite design, constellation beam patterns through a series of time.

### Simulation grid explanation

Currently, 5 dimensions are used:
1. telescope pointings/sky grid cells
2. epochs (separate observations)
3. number of satellites
4. iterations: the subintegrations during an observation
5. (under development) individual beams

